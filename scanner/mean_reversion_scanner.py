#!/usr/bin/env python3
"""
Mean-reversion $5-wide vertical spread scanner.

Strategy reference: strategy/mean-reversion-spreads.html (Strategy Lab).

Data sources (free, keyless — no paid or rate-limited APIs):
  * Yahoo Finance public endpoints via `yfinance` — daily bars + live option
    chains. No API key, no signup.
  * Stooq daily CSV — automatic fallback for price history if Yahoo is
    unreachable. Prices only, so the scanner degrades to signal-only mode
    (no spread construction) on this path.

Setup:
    pip install yfinance pandas numpy

Usage:
    python mean_reversion_scanner.py                          # full scan
    python mean_reversion_scanner.py --side bull              # bullish only
    python mean_reversion_scanner.py --profile balanced       # one profile
    python mean_reversion_scanner.py --tickers AAPL,MSFT,AVGO # custom universe
    python mean_reversion_scanner.py --csv results.csv        # save output

Entry checklist implemented (all required):
  Bullish:  close > SMA200, RSI(2) < 5, close < lower Bollinger(20,2),
            5-day return z-score < -1.5, HV-rank >= 30 (IV-rank proxy)
  Bearish:  mirror conditions below SMA200.

Spread construction (per signal ticker, from the live option chain):
  Expiry 30-45 DTE; short strike at the profile delta (10/30/50), long strike
  $5 farther out; credit from leg mids; POP via Black-Scholes N(d2) using the
  chain's own implied vol; EV = POP*credit - (1-POP)*(width-credit).

This scanner finds candidates. Liquidity (OI/spread width) and the earnings
calendar must still be checked before entry — see the strategy page.
"""

import argparse
import csv
import io
import math
import sys
import urllib.request
from datetime import datetime, timezone

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("Missing dependencies. Run: pip install yfinance pandas numpy")

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False

# ── Configuration ────────────────────────────────────────────────────────────

WIDTH = 5.0          # spread width in dollars
RISK_FREE = 0.04     # annualized, for Black-Scholes
DTE_MIN, DTE_MAX = 30, 45

# Profile name -> target |delta| of the short strike
PROFILES = {
    "conservative": 0.10,   # ~90% structural POP, ~12% max ROI
    "balanced":     0.30,   # ~70% structural POP, ~49% max ROI  (recommended)
    "aggressive":   0.50,   # ~50% structural POP, ~100% max ROI (literal spec)
}

# Liquid, optionable, $1-strike-increment large caps. Edit freely.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO", "MRVL",
    "QCOM", "MU", "AMAT", "LRCX", "KLAC", "TSM", "INTC", "TXN", "ADI",
    "CRM", "NOW", "ORCL", "ADBE", "PANW", "CRWD", "NET", "DDOG", "SNOW",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO",
    "XOM", "CVX", "COP", "SLB",
    "HD", "LOW", "COST", "WMT", "TGT", "MCD", "NKE", "SBUX",
    "CAT", "DE", "GE", "HON", "BA", "UNP",
    "DIS", "NFLX", "TSLA", "UBER", "ABNB",
    "SPY", "QQQ", "IWM", "DIA",
]

# ── Indicators ───────────────────────────────────────────────────────────────

def rsi(series: pd.Series, n: int = 2) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0)


def compute_signals(close: pd.Series) -> dict | None:
    """Evaluate the entry checklist on a daily close series (>= 1y of data)."""
    if close.dropna().shape[0] < 220:
        return None
    c = close.dropna()

    sma200 = c.rolling(200).mean().iloc[-1]
    rsi2 = rsi(c, 2).iloc[-1]

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower_bb = (mid - 2.0 * sd).iloc[-1]
    upper_bb = (mid + 2.0 * sd).iloc[-1]

    ret5 = c.pct_change(5)
    z5 = (ret5.iloc[-1] - ret5.mean()) / ret5.std(ddof=0)

    # Historical-vol rank as a free IV-rank proxy (20d realized vol percentile
    # over 1y). When the chain is available we also have true IV per strike.
    logret = np.log(c / c.shift(1))
    hv20 = logret.rolling(20).std(ddof=0) * math.sqrt(252)
    hv_window = hv20.dropna().iloc[-252:]
    hv_rank = float((hv_window <= hv_window.iloc[-1]).mean() * 100)

    px = float(c.iloc[-1])
    bull = (px > sma200) and (rsi2 < 5) and (px < lower_bb) and (z5 < -1.5) \
        and (hv_rank >= 30)
    bear = (px < sma200) and (rsi2 > 95) and (px > upper_bb) and (z5 > 1.5) \
        and (hv_rank >= 30)

    if not (bull or bear):
        return None
    return {
        "side": "bull" if bull else "bear",
        "price": px,
        "rsi2": float(rsi2),
        "z5": float(z5),
        "hv_rank": hv_rank,
        "hv20": float(hv20.iloc[-1]),
    }

# ── Black-Scholes helpers (math-stdlib only, no scipy) ───────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1_d2(spot, strike, iv, t_years, r=RISK_FREE):
    if iv <= 0 or t_years <= 0 or spot <= 0 or strike <= 0:
        return None, None
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) \
        / (iv * math.sqrt(t_years))
    return d1, d1 - iv * math.sqrt(t_years)


def put_delta(spot, strike, iv, t_years):
    d1, _ = bs_d1_d2(spot, strike, iv, t_years)
    return None if d1 is None else _norm_cdf(d1) - 1.0


def call_delta(spot, strike, iv, t_years):
    d1, _ = bs_d1_d2(spot, strike, iv, t_years)
    return None if d1 is None else _norm_cdf(d1)


def pop_short_put(spot, strike, iv, t_years):
    """P(S_T > K) — probability a short put expires worthless."""
    _, d2 = bs_d1_d2(spot, strike, iv, t_years)
    return None if d2 is None else _norm_cdf(d2)


def pop_short_call(spot, strike, iv, t_years):
    """P(S_T < K) — probability a short call expires worthless."""
    _, d2 = bs_d1_d2(spot, strike, iv, t_years)
    return None if d2 is None else _norm_cdf(-d2)

# ── Price data (Yahoo primary, Stooq fallback) ───────────────────────────────

def fetch_prices_yahoo(tickers: list[str]) -> pd.DataFrame | None:
    if not HAVE_YF:
        return None
    try:
        data = yf.download(tickers, period="1y", interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        if data.empty:
            return None
        closes = data["Close"]
        if isinstance(closes, pd.Series):           # single ticker
            closes = closes.to_frame(tickers[0])
        return closes
    except Exception as exc:
        print(f"  Yahoo price download failed: {exc}", file=sys.stderr)
        return None


def fetch_prices_stooq(tickers: list[str]) -> pd.DataFrame | None:
    """Free CSV endpoint, one request per ticker. US tickers use '.us'."""
    frames = {}
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode()
            df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"],
                             index_col="Date")
            if not df.empty:
                frames[t] = df["Close"].iloc[-260:]
        except Exception:
            continue
    if not frames:
        return None
    return pd.DataFrame(frames)

# ── Spread construction from the live chain ──────────────────────────────────

def pick_expiry(expiries: list[str]) -> tuple[str, int] | None:
    today = datetime.now(timezone.utc).date()
    best, best_dist = None, 10**9
    for e in expiries:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if DTE_MIN <= dte <= DTE_MAX:
            return e, dte
        dist = min(abs(dte - DTE_MIN), abs(dte - DTE_MAX))
        if 20 <= dte <= 60 and dist < best_dist:
            best, best_dist = (e, dte), dist
    return best


def mid_price(row) -> float | None:
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    last = float(row.get("lastPrice") or 0)
    return last if last > 0 else None


def build_spreads(ticker: str, sig: dict, profiles: dict) -> list[dict]:
    """Construct $5-wide vertical candidates from the Yahoo option chain."""
    results = []
    if not HAVE_YF:
        return results
    try:
        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])
        picked = pick_expiry(expiries)
        if not picked:
            return results
        expiry, dte = picked
        chain = tk.option_chain(expiry)
    except Exception:
        return results

    t_years = dte / 365.0
    spot = sig["price"]
    legs = chain.puts if sig["side"] == "bull" else chain.calls
    delta_fn = put_delta if sig["side"] == "bull" else call_delta
    pop_fn = pop_short_put if sig["side"] == "bull" else pop_short_call

    legs = legs.dropna(subset=["strike"]).set_index("strike", drop=False)

    for name, target in profiles.items():
        best_row, best_diff = None, 10**9
        for _, row in legs.iterrows():
            iv = float(row.get("impliedVolatility") or 0)
            if iv <= 0.01:
                iv = sig["hv20"]            # fall back to realized vol
            d = delta_fn(spot, float(row["strike"]), iv, t_years)
            if d is None:
                continue
            diff = abs(abs(d) - target)
            if diff < best_diff:
                best_row, best_diff, best_iv = row, diff, iv
        if best_row is None or best_diff > 0.12:
            continue

        short_k = float(best_row["strike"])
        long_k = short_k - WIDTH if sig["side"] == "bull" else short_k + WIDTH
        if long_k not in legs.index:
            # accept the nearest strike within $4.50-$5.50 and keep true width
            near = legs.index[(abs(legs.index - long_k) <= 0.5)]
            if len(near) == 0:
                continue
            long_k = float(near[0])
        width = abs(short_k - long_k)

        m_short = mid_price(best_row)
        m_long = mid_price(legs.loc[long_k])
        if m_short is None or m_long is None:
            continue
        credit = round(m_short - m_long, 2)
        if credit <= 0.05:
            continue

        pop = pop_fn(spot, short_k, best_iv, t_years)
        max_loss = width - credit
        ev = pop * credit - (1.0 - pop) * max_loss
        results.append({
            "ticker": ticker, "side": sig["side"], "profile": name,
            "expiry": expiry, "dte": dte, "short": short_k, "long": long_k,
            "credit": credit, "max_loss": round(max_loss, 2),
            "max_roi": round(100 * credit / max_loss, 1),
            "pop": round(100 * pop, 1), "ev": round(ev, 2),
            "rsi2": round(sig["rsi2"], 1), "z5": round(sig["z5"], 2),
            "hv_rank": round(sig["hv_rank"]),
        })
    return results

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--side", choices=["bull", "bear", "both"], default="both")
    ap.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    ap.add_argument("--tickers", help="comma-separated custom universe")
    ap.add_argument("--csv", help="write results to this CSV path")
    args = ap.parse_args()

    universe = ([t.strip().upper() for t in args.tickers.split(",")]
                if args.tickers else UNIVERSE)
    profiles = (PROFILES if args.profile == "all"
                else {args.profile: PROFILES[args.profile]})

    print(f"Scanning {len(universe)} tickers "
          f"(side={args.side}, profiles={', '.join(profiles)}) ...")

    closes = fetch_prices_yahoo(universe)
    chain_ok = closes is not None
    if closes is None:
        print("Yahoo unavailable — falling back to Stooq (signal-only mode).")
        closes = fetch_prices_stooq(universe)
    if closes is None:
        print("No price data available from any free source. Use the "
              "thinkorswim scan in scanner/MeanReversionSpreads.ts instead.")
        return 1

    signals = {}
    for t in closes.columns:
        sig = compute_signals(closes[t])
        if sig and (args.side in ("both", sig["side"])):
            signals[t] = sig

    if not signals:
        print("No tickers pass the full entry checklist today. "
              "(That is normal — extremes are rare by design.)")
        return 0

    print(f"\nSignals: " + ", ".join(
        f"{t} ({s['side']}, RSI2={s['rsi2']:.1f}, z={s['z5']:.2f})"
        for t, s in signals.items()))

    rows = []
    if chain_ok:
        for t, sig in signals.items():
            rows.extend(build_spreads(t, sig, profiles))
    if not rows:
        print("\nSignal-only mode (no chain data): pick strikes manually at "
              "your profile delta from your broker's chain.")
        return 0

    rows.sort(key=lambda r: r["ev"], reverse=True)

    hdr = (f"{'TICKER':<7}{'SIDE':<6}{'PROFILE':<13}{'EXPIRY':<12}{'DTE':>4}"
           f"{'SHORT':>8}{'LONG':>8}{'CREDIT':>8}{'MAXROI':>8}{'POP':>7}"
           f"{'EV':>7}{'RSI2':>6}{'Z5':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['ticker']:<7}{r['side']:<6}{r['profile']:<13}"
              f"{r['expiry']:<12}{r['dte']:>4}{r['short']:>8.1f}"
              f"{r['long']:>8.1f}{r['credit']:>8.2f}{r['max_roi']:>7.1f}%"
              f"{r['pop']:>6.1f}%{r['ev']:>+7.2f}{r['rsi2']:>6.1f}"
              f"{r['z5']:>7.2f}")

    print("\nReminders: verify OI >= 500 and leg bid/ask <= $0.15; check the "
          "earnings calendar; exits per the strategy page (80% profit take, "
          "2x credit stop, 21 DTE / 30-day time exit).")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
