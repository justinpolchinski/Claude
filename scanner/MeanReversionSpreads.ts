# =============================================================================
# MeanReversionSpreads — thinkorswim study + Stock Hacker scan code
# Companion to strategy/mean-reversion-spreads.html (Strategy Lab)
#
# Strategy: $5-wide vertical credit spreads entered at mean-reversion extremes.
#   Bullish -> sell bull put spread   (close > SMA200, RSI(2) < 5,
#                                      close < lower Bollinger, z-score < -1.5)
#   Bearish -> sell bear call spread  (mirror, below SMA200)
#
# thinkScript cannot price spreads inside a scan, so the workflow is:
#   1. Stock Hacker scan (snippets at the bottom) finds signal tickers.
#   2. This chart study confirms the setup and labels the short-strike zone
#      and IV Rank.
#   3. Build the $5-wide spread from the option chain at your profile delta:
#        Conservative 10Δ (~90% POP, ~12% ROI)
#        Balanced     30Δ (~70% POP, ~49% ROI)  <- recommended
#        Aggressive   50Δ (~50% POP, ~100% ROI)
#   4. Exits: 80% profit take, 2x-credit stop, 21 DTE / 30-day time exit.
# =============================================================================

# ---------- CHART STUDY (Studies -> Edit Studies -> Create) ----------

declare upper;

input rsiLength      = 2;
input rsiOversold    = 5;
input rsiOverbought  = 95;
input bbLength       = 20;
input bbDev          = 2.0;
input trendLength    = 200;
input zLookback      = 252;
input zThreshold     = 1.5;
input ivRankMin      = 30;

# --- Indicators ---
def sma200 = Average(close, trendLength);
def rsi2   = RSI(price = close, length = rsiLength);

def bbMid  = Average(close, bbLength);
def bbSd   = StDev(close, bbLength);
def lowBB  = bbMid - bbDev * bbSd;
def upBB   = bbMid + bbDev * bbSd;

# 5-day return z-score vs ~1y distribution
def ret5   = close / close[5] - 1;
def zScore = (ret5 - Average(ret5, zLookback)) / StDev(ret5, zLookback);

# IV Rank (1y) — works on daily charts for optionable symbols
def iv     = imp_volatility();
def ivLo   = Lowest(iv, zLookback);
def ivHi   = Highest(iv, zLookback);
def ivRank = if ivHi != ivLo then 100 * (iv - ivLo) / (ivHi - ivLo)
             else Double.NaN;

# --- Signals ---
def bullSignal = close > sma200
             and rsi2 < rsiOversold
             and close < lowBB
             and zScore < -zThreshold
             and ivRank >= ivRankMin;

def bearSignal = close < sma200
             and rsi2 > rsiOverbought
             and close > upBB
             and zScore > zThreshold
             and ivRank >= ivRankMin;

# --- Plots ---
plot BullEntry = bullSignal;
BullEntry.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_UP);
BullEntry.SetDefaultColor(Color.GREEN);
BullEntry.SetLineWeight(3);

plot BearEntry = bearSignal;
BearEntry.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_DOWN);
BearEntry.SetDefaultColor(Color.RED);
BearEntry.SetLineWeight(3);

# Short-strike zone guide: the band the move would have to crash through.
plot ShortPutZone = lowBB;
ShortPutZone.SetDefaultColor(Color.DARK_GREEN);
ShortPutZone.SetStyle(Curve.SHORT_DASH);

plot ShortCallZone = upBB;
ShortCallZone.SetDefaultColor(Color.DARK_RED);
ShortCallZone.SetStyle(Curve.SHORT_DASH);

plot Trend = sma200;
Trend.SetDefaultColor(Color.GRAY);

# --- Labels ---
AddLabel(yes, "RSI(2): " + Round(rsi2, 1),
         if rsi2 < rsiOversold then Color.GREEN
         else if rsi2 > rsiOverbought then Color.RED
         else Color.GRAY);
AddLabel(yes, "Z(5d): " + Round(zScore, 2),
         if AbsValue(zScore) > zThreshold then Color.YELLOW else Color.GRAY);
AddLabel(!IsNaN(ivRank), "IV Rank: " + Round(ivRank, 0),
         if ivRank >= ivRankMin then Color.CYAN else Color.GRAY);
AddLabel(bullSignal,
         "BULL PUT SPREAD: short strike at/below " + Round(lowBB, 0) +
         " (pick 10/30/50 delta), long $5 lower, 30-45 DTE", Color.GREEN);
AddLabel(bearSignal,
         "BEAR CALL SPREAD: short strike at/above " + Round(upBB, 0) +
         " (pick 10/30/50 delta), long $5 higher, 30-45 DTE", Color.RED);

# =============================================================================
# STOCK HACKER SCAN SNIPPETS
# Scan tab -> Stock Hacker -> Add filter -> Study -> pencil icon ->
# thinkScript Editor -> delete contents, paste ONE block below.
# Recommended base filters: Stock, Last >= 20, Volume >= 1,000,000,
# and under Options: "Has options: Yes".
# Note: imp_volatility() is not reliable inside scans on all accounts, so the
# IV Rank condition is applied on the chart study, not in the scan.
# =============================================================================

# ---- BULLISH SCAN (paste alone) ----
# def sma200 = Average(close, 200);
# def rsi2   = RSI(price = close, length = 2);
# def bbMid  = Average(close, 20);
# def bbSd   = StDev(close, 20);
# def ret5   = close / close[5] - 1;
# def z      = (ret5 - Average(ret5, 252)) / StDev(ret5, 252);
# plot scan  = close > sma200 and rsi2 < 5
#          and close < bbMid - 2.0 * bbSd and z < -1.5;

# ---- BEARISH SCAN (paste alone) ----
# def sma200 = Average(close, 200);
# def rsi2   = RSI(price = close, length = 2);
# def bbMid  = Average(close, 20);
# def bbSd   = StDev(close, 20);
# def ret5   = close / close[5] - 1;
# def z      = (ret5 - Average(ret5, 252)) / StDev(ret5, 252);
# plot scan  = close < sma200 and rsi2 > 95
#          and close > bbMid + 2.0 * bbSd and z > 1.5;
