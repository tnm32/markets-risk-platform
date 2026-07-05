"""
Page 4: Risk Analytics
======================
Covers: Historical VaR, Parametric VaR, Expected Shortfall (CVaR)

Interview relevance
-------------------
- S&T: Every trading desk has a VaR limit. You need to explain what VaR means,
  what its weaknesses are, and why ES was added to Basel III/IV.
- Risk management roles: VaR and ES are the two headline metrics in market risk.
  Knowing *why* parametric VaR assumes normality — and why that breaks down
  in a crisis — is exactly what interviewers probe.
- Asset management: Risk-adjusted returns, drawdown analysis, and tail risk
  all tie back to these concepts.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import numpy as np                    # numerical arrays and maths
import pandas as pd                   # DataFrames (tables of data)
import plotly.graph_objects as go     # interactive charts
import plotly.express as px           # simpler chart shorthand
import yfinance as yf                 # live price data from Yahoo Finance
import streamlit as st                # the web-app framework
from scipy import stats               # statistical distributions (normal, etc.)
from datetime import date, timedelta  # date arithmetic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Analytics",
    page_icon="⚠️",
    layout="wide",
)

# ── Title block ───────────────────────────────────────────────────────────────
st.title("⚠️ Risk Analytics")
st.markdown(
    """
    Market risk measured three ways: **Historical VaR**, **Parametric VaR**, 
    and **Expected Shortfall (ES/CVaR)**.  
    All three are standard outputs on any sell-side risk desk and are embedded 
    in the Basel III/IV regulatory framework.
    """
)

# ── Concept explainer (collapsed by default so the page stays clean) ──────────
with st.expander("📚 Finance concepts — read before your interview"):
    st.markdown(
        """
        ### What is VaR?
        **Value at Risk (VaR)** answers: *"What is the most I can lose over a given 
        horizon, at a given confidence level, under normal market conditions?"*

        A 1-day 95 % VaR of £1 m means: *on 95 % of days I expect to lose less than 
        £1 m. On the remaining 5 % of days I may lose more.*

        ### Three ways to calculate it

        | Method | How it works | Key assumption |
        |---|---|---|
        | **Historical** | Sort actual past returns worst→best; read off the percentile | No distribution assumed — history is the model |
        | **Parametric (variance-covariance)** | Fit a normal distribution to returns; use z-score × σ | Returns are normally distributed |
        | **Monte Carlo** | Simulate thousands of random paths; read off the percentile | Whatever distribution you choose to simulate |

        We implement the first two here; Monte Carlo appears in Page 3's efficient frontier.

        ### Why does it matter which method you use?
        - **Parametric VaR underestimates tail risk** because real returns have 
          *fat tails* (leptokurtosis) — extreme moves happen far more often than 
          a normal distribution predicts. This is why banks lost far more in 2008 
          than their VaR models suggested.
        - **Historical VaR is backward-looking** — it cannot capture a crisis regime 
          that has never happened before (e.g. COVID-19 in Feb–Mar 2020 was outside 
          most historical windows).

        ### Expected Shortfall (ES / CVaR)
        ES fixes VaR's most-criticised flaw: VaR tells you *where* the loss threshold 
        is but says nothing about *how bad* losses beyond that threshold can get.

        **ES = average loss in the worst (1 − c)% of scenarios**

        e.g. 95 % ES = average loss on the 5 % worst days.

        ES is **sub-additive** (diversification always reduces it) and is the 
        metric required under **FRTB (Basel IV)** for regulatory capital, replacing 
        VaR as the headline number from 2025.

        ### Interview hook
        > *"Our model showed a 1-day 99 % VaR of $50 m, but we still lost $200 m 
        > on one day in 2008. How is that possible?"*  
        > VaR is a threshold, not a worst-case. ES would have given a better picture 
        > of the *average* loss beyond that threshold. And the historical window 
        > pre-2008 simply didn't contain 2008-style volatility.
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Portfolio inputs
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. Build your portfolio")

# ---------- ticker input -----------------------------------------------------
# st.text_input returns whatever the user has typed as a plain Python string.
default_tickers = "AAPL, MSFT, JPM, GS, SPY"
raw_input = st.text_input(
    "Enter tickers (comma-separated)",
    value=default_tickers,
    help="Use Yahoo Finance tickers, e.g. AAPL, TSLA, ^GSPC",
)

# Split the string on commas, strip spaces, convert to upper-case.
# The list comprehension [ x for x in iterable ] is Python shorthand for a loop
# that builds a new list.
tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

# ---------- date range -------------------------------------------------------
col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input(
        "Start date",
        value=date.today() - timedelta(days=3 * 365),  # default: 3 years back
    )
with col_d2:
    end_date = st.date_input("End date", value=date.today())

# ---------- weights ----------------------------------------------------------
st.subheader("Portfolio weights")
st.caption(
    "Adjust the slider for each asset. Weights are automatically normalised to 100 %."
)

# We create one slider per ticker.
# st.slider returns a float between the min and max you specify.
raw_weights = {}
slider_cols = st.columns(len(tickers))  # one column per ticker
for i, ticker in enumerate(tickers):
    with slider_cols[i]:
        raw_weights[ticker] = st.slider(
            ticker,
            min_value=0,
            max_value=100,
            value=100 // len(tickers),  # default: equal weight
            step=1,
        )

# Normalise: divide each weight by the total so they sum to 1.0
total_w = sum(raw_weights.values())
if total_w == 0:
    st.error("All weights are zero — please assign at least one non-zero weight.")
    st.stop()  # halt execution of this script here

weights = np.array([raw_weights[t] / total_w for t in tickers])

# Show the normalised weights as a small table
weight_df = pd.DataFrame(
    {"Ticker": tickers, "Weight (%)": [f"{w*100:.1f}" for w in weights]}
)
st.dataframe(weight_df, hide_index=True, use_container_width=False)

# ---------- VaR parameters ---------------------------------------------------
st.subheader("Risk parameters")
col_c, col_h = st.columns(2)
with col_c:
    # Confidence level: 0.95 means 95th percentile VaR
    confidence = st.select_slider(
        "Confidence level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{x*100:.0f}%",
    )
with col_h:
    horizon_days = st.select_slider(
        "VaR horizon (days)",
        options=[1, 5, 10],
        value=1,
        help="Basel uses 10-day horizon for market risk capital.",
    )

portfolio_value = st.number_input(
    "Portfolio value (£)",
    min_value=1_000,
    max_value=100_000_000,
    value=1_000_000,
    step=10_000,
    help="Used to convert percentage VaR into a pound-sterling loss figure.",
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Data download and return calculation
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Downloading prices…")
def load_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.
    Returns a DataFrame where each column is one ticker.

    @st.cache_data tells Streamlit: if this function is called again with the
    *same arguments*, skip the download and return the cached result.
    That keeps the app fast when you tweak sliders but haven't changed tickers.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns a MultiIndex DataFrame when multiple tickers are requested.
    # We want only the "Close" column for each ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices.dropna(how="all", inplace=True)  # remove days where everything is NaN
    return prices


prices = load_prices(tickers, str(start_date), str(end_date))

# Drop tickers that failed to download (all NaN column)
valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().any()]
if not valid_tickers:
    st.error("Could not download data for any of the tickers. Check your spelling.")
    st.stop()

prices = prices[valid_tickers]

# Re-align weights to only valid tickers
weights = np.array([raw_weights[t] / total_w for t in valid_tickers])

# ---------- Daily returns ----------------------------------------------------
# pct_change() computes (today - yesterday) / yesterday for every column.
# dropna() removes the first row (which becomes NaN because there is no
# "yesterday" for day 1).
returns = prices.pct_change().dropna()

# ---------- Portfolio daily return series ------------------------------------
# Matrix multiplication: (N_days × N_assets) @ (N_assets,) → (N_days,)
# Each day's portfolio return = sum of (weight × asset return) across assets.
# The @ operator is Python's matrix multiplication operator (introduced in PEP 465).
portfolio_returns = returns[valid_tickers] @ weights

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VaR calculations
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Risk metrics")

# ---------- Helper: scale 1-day VaR to multi-day horizon ─────────────────────
# The "square-root-of-time" rule:  VaR_T = VaR_1 × √T
# This assumes returns are i.i.d. (independent and identically distributed),
# which is an approximation but is the Basel standard.
sqrt_t = np.sqrt(horizon_days)

# ── A. Historical VaR ────────────────────────────────────────────────────────
# Sort all portfolio returns and take the (1 - confidence) percentile.
# np.percentile(arr, q) returns the q-th percentile of arr.
# e.g. at 95% confidence we want the 5th percentile (worst 5% of days).
hist_var_pct = -np.percentile(portfolio_returns, (1 - confidence) * 100)
hist_var_pct_horizon = hist_var_pct * sqrt_t
hist_var_gbp = hist_var_pct_horizon * portfolio_value

# ── B. Parametric VaR ────────────────────────────────────────────────────────
# Assumptions: returns ~ Normal(μ, σ²)
# VaR = μ − z × σ   where z = the confidence-level z-score
#
# stats.norm.ppf(p) is the "percent point function" (inverse CDF) of the
# standard normal distribution.  ppf(0.05) ≈ −1.645, ppf(0.01) ≈ −2.326.
mu = portfolio_returns.mean()       # daily mean return
sigma = portfolio_returns.std()     # daily standard deviation
z_score = stats.norm.ppf(1 - confidence)   # negative number, e.g. −1.645

param_var_pct = -(mu + z_score * sigma)    # negated so it reads as a positive loss
param_var_pct_horizon = param_var_pct * sqrt_t
param_var_gbp = param_var_pct_horizon * portfolio_value

# ── C. Expected Shortfall (Historical) ───────────────────────────────────────
# ES = average of all returns *worse* than the VaR threshold.
# Boolean mask: portfolio_returns < -hist_var_pct selects only the tail days.
tail_returns = portfolio_returns[portfolio_returns < -hist_var_pct]
if len(tail_returns) == 0:
    hist_es_pct = hist_var_pct   # fallback if window is tiny
else:
    hist_es_pct = -tail_returns.mean()

hist_es_pct_horizon = hist_es_pct * sqrt_t
hist_es_gbp = hist_es_pct_horizon * portfolio_value

# ---------- Display metric cards ---------------------------------------------
m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label=f"Historical VaR ({confidence*100:.0f}%, {horizon_days}d)",
        value=f"£{hist_var_gbp:,.0f}",
        delta=f"{hist_var_pct_horizon*100:.2f}% of portfolio",
        delta_color="inverse",   # red for losses
    )
    st.caption("Based on actual return distribution — no normality assumption.")

with m2:
    st.metric(
        label=f"Parametric VaR ({confidence*100:.0f}%, {horizon_days}d)",
        value=f"£{param_var_gbp:,.0f}",
        delta=f"{param_var_pct_horizon*100:.2f}% of portfolio",
        delta_color="inverse",
    )
    st.caption("Assumes normally distributed returns. Underestimates fat tails.")

with m3:
    st.metric(
        label=f"Expected Shortfall ({confidence*100:.0f}%, {horizon_days}d)",
        value=f"£{hist_es_gbp:,.0f}",
        delta=f"{hist_es_pct_horizon*100:.2f}% of portfolio",
        delta_color="inverse",
    )
    st.caption("Average loss beyond VaR. Required under Basel IV / FRTB.")

# Comparison note
st.info(
    f"**ES/VaR ratio: {hist_es_pct/hist_var_pct:.2f}x** — "
    "a ratio above 1.3× in normal markets (or 2×+ in crises) signals heavy tail risk. "
    "Regulators watch this closely."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Return distribution chart
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Return distribution")

# We plot a histogram of actual returns alongside the fitted normal curve
# so you can visually see whether fat tails are present.

fig_dist = go.Figure()

# --- Histogram of actual returns ---
fig_dist.add_trace(
    go.Histogram(
        x=portfolio_returns * 100,   # convert to percentage for readability
        nbinsx=60,
        name="Actual returns",
        marker_color="#3b82f6",
        opacity=0.6,
        histnorm="probability density",   # normalise so it's comparable to the PDF
    )
)

# --- Fitted normal distribution (PDF) ---
# np.linspace(a, b, n) creates n evenly-spaced points between a and b.
x_range = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 300)
normal_pdf = stats.norm.pdf(x_range, loc=mu, scale=sigma)

fig_dist.add_trace(
    go.Scatter(
        x=x_range * 100,
        y=normal_pdf / 100,   # scale denominator to match percentage x-axis
        mode="lines",
        name="Normal fit",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    )
)

# --- VaR threshold lines ---
# add_vline draws a vertical line at a given x value.
fig_dist.add_vline(
    x=-hist_var_pct * 100,
    line=dict(color="#ef4444", width=2, dash="solid"),
    annotation_text=f"Hist VaR {confidence*100:.0f}%",
    annotation_position="top right",
)
fig_dist.add_vline(
    x=-param_var_pct * 100,
    line=dict(color="#f97316", width=2, dash="dot"),
    annotation_text=f"Param VaR {confidence*100:.0f}%",
    annotation_position="top left",
)

fig_dist.update_layout(
    title="Portfolio daily return distribution",
    xaxis_title="Daily return (%)",
    yaxis_title="Probability density",
    legend=dict(orientation="h", y=-0.2),
    template="plotly_dark",
    height=420,
    bargap=0.05,
)

st.plotly_chart(fig_dist, use_container_width=True)

# --- Tail statistics (kurtosis tells us about fat tails) ---
kurt = portfolio_returns.kurt()      # excess kurtosis; normal dist = 0
skew = portfolio_returns.skew()

col_k1, col_k2 = st.columns(2)
with col_k1:
    st.metric("Excess kurtosis", f"{kurt:.2f}", help="Normal distribution = 0. Positive = fatter tails than normal.")
with col_k2:
    st.metric("Skewness", f"{skew:.2f}", help="Normal distribution = 0. Negative = left-skewed (more big negative days).")

if kurt > 1:
    st.warning(
        f"Excess kurtosis of {kurt:.2f} confirms **fat tails**: extreme returns "
        "occur more often than a normal distribution predicts. "
        "This is why Parametric VaR may underestimate your true tail risk."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Rolling VaR through time
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. Rolling VaR over time")
st.markdown(
    "A static VaR number misses *when* your portfolio was most dangerous. "
    "Rolling VaR shows how risk evolved — spikes coincide with market stress events."
)

rolling_window = st.slider(
    "Rolling window (trading days)",
    min_value=21,       # ~1 month
    max_value=252,      # ~1 year
    value=63,           # ~3 months
    step=21,
    help="Shorter windows react faster to changing volatility; longer windows are smoother.",
)

# rolling(n) creates a sliding window of n observations.
# .apply(func) applies a custom function to each window.
# lambda r: ... defines a small anonymous function inline.
rolling_hist_var = portfolio_returns.rolling(rolling_window).apply(
    lambda r: -np.percentile(r, (1 - confidence) * 100)
)

rolling_param_var = portfolio_returns.rolling(rolling_window).apply(
    lambda r: -(r.mean() + stats.norm.ppf(1 - confidence) * r.std())
)

fig_roll = go.Figure()

fig_roll.add_trace(
    go.Scatter(
        x=rolling_hist_var.index,
        y=rolling_hist_var * 100,
        name="Rolling Historical VaR",
        line=dict(color="#ef4444", width=1.5),
    )
)

fig_roll.add_trace(
    go.Scatter(
        x=rolling_param_var.index,
        y=rolling_param_var * 100,
        name="Rolling Parametric VaR",
        line=dict(color="#f97316", width=1.5, dash="dot"),
    )
)

# Shade the actual daily losses for visual context
fig_roll.add_trace(
    go.Scatter(
        x=portfolio_returns.index,
        y=-portfolio_returns * 100,   # flip sign: positive = a loss
        name="Daily P&L (loss = positive)",
        line=dict(color="#94a3b8", width=0.5),
        opacity=0.4,
    )
)

fig_roll.update_layout(
    title=f"{rolling_window}-day rolling {confidence*100:.0f}% VaR",
    xaxis_title="Date",
    yaxis_title="Loss (%)",
    legend=dict(orientation="h", y=-0.2),
    template="plotly_dark",
    height=400,
)

st.plotly_chart(fig_roll, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VaR backtesting (breaches)
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. VaR backtesting")
st.markdown(
    """
    **Backtesting** checks whether the VaR model performed as promised.  
    A 95 % VaR should be *breached* (actual loss > VaR) on roughly 5 % of days.  
    Too many breaches → model underestimates risk. Too few → model is too conservative 
    (capital is over-allocated).  
    Regulators (Basel) require banks to backtest their VaR models daily.
    """
)

# A breach occurs when the actual portfolio loss exceeds the VaR estimate.
# We use rolling historical VaR as the "live" estimate.
# .shift(1) moves the VaR series forward by one day so we compare yesterday's
# VaR estimate to today's actual return (no look-ahead bias).
actual_losses = -portfolio_returns         # positive = loss
var_estimates = rolling_hist_var.shift(1)  # yesterday's estimate

# Drop rows where rolling VaR hasn't yet been calculated (the first window)
backtest_df = pd.DataFrame(
    {"Actual loss (%)": actual_losses * 100, "VaR estimate (%)": var_estimates * 100}
).dropna()

breaches = backtest_df[backtest_df["Actual loss (%)"] > backtest_df["VaR estimate (%)"]]
breach_rate = len(breaches) / len(backtest_df) * 100
expected_rate = (1 - confidence) * 100

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    st.metric("Total trading days", f"{len(backtest_df):,}")
with col_b2:
    st.metric("VaR breaches", f"{len(breaches)}", help="Days where actual loss exceeded VaR.")
with col_b3:
    st.metric(
        "Breach rate",
        f"{breach_rate:.1f}%",
        delta=f"Expected: {expected_rate:.1f}%",
        delta_color="off",
    )

if breach_rate > expected_rate * 1.5:
    st.error(
        f"Breach rate of {breach_rate:.1f}% significantly exceeds the expected "
        f"{expected_rate:.1f}%. Under Basel rules this could trigger a capital surcharge."
    )
elif breach_rate < expected_rate * 0.5:
    st.warning(
        f"Breach rate of {breach_rate:.1f}% is well below the expected {expected_rate:.1f}%. "
        "Your VaR model may be too conservative — capital is being over-allocated."
    )
else:
    st.success(
        f"Breach rate of {breach_rate:.1f}% is close to the expected {expected_rate:.1f}%. "
        "The model is well-calibrated."
    )

# Scatter: actual losses vs VaR, highlighting breaches
fig_bt = go.Figure()

# Non-breach days
non_breach = backtest_df[backtest_df["Actual loss (%)"] <= backtest_df["VaR estimate (%)"]]
fig_bt.add_trace(
    go.Scatter(
        x=non_breach.index,
        y=non_breach["Actual loss (%)"],
        mode="markers",
        name="No breach",
        marker=dict(color="#3b82f6", size=3, opacity=0.5),
    )
)

# VaR line
fig_bt.add_trace(
    go.Scatter(
        x=backtest_df.index,
        y=backtest_df["VaR estimate (%)"],
        mode="lines",
        name="VaR estimate",
        line=dict(color="#f59e0b", width=1.5),
    )
)

# Breach days — plotted on top in red
fig_bt.add_trace(
    go.Scatter(
        x=breaches.index,
        y=breaches["Actual loss (%)"],
        mode="markers",
        name="Breach",
        marker=dict(color="#ef4444", size=6, symbol="x"),
    )
)

fig_bt.update_layout(
    title=f"VaR backtest — {confidence*100:.0f}% confidence, {rolling_window}-day rolling window",
    xaxis_title="Date",
    yaxis_title="Loss (% of portfolio)",
    legend=dict(orientation="h", y=-0.2),
    template="plotly_dark",
    height=420,
)

st.plotly_chart(fig_bt, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Maximum drawdown
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. Drawdown analysis")
st.markdown(
    """
    **Maximum Drawdown (MDD)** measures the largest peak-to-trough decline in 
    portfolio value. It answers: *"If I invested at the worst possible moment, 
    how much of my money would I have lost at the worst point?"*  
    Asset managers watch MDD closely — many mandates have hard drawdown limits 
    (e.g. a fund must de-risk if drawdown exceeds 15 %).
    """
)

# Compute portfolio cumulative returns
# cumprod() = cumulative product: (1+r1) × (1+r2) × … gives a £1 growth curve.
cum_returns = (1 + portfolio_returns).cumprod()

# Running maximum: at each point, what was the highest value ever reached?
# expanding().max() = look at all data up to and including today, take the max.
running_max = cum_returns.expanding().max()

# Drawdown at each point = how far below the peak are we right now?
drawdown = (cum_returns - running_max) / running_max

max_drawdown = drawdown.min()   # the most negative value = deepest drawdown

st.metric(
    "Maximum drawdown",
    f"{max_drawdown*100:.2f}%",
    help="Peak-to-trough decline in portfolio value over the period.",
    delta_color="inverse",
)

fig_dd = go.Figure()

# Shade the drawdown area — fill="tozeroy" fills between the line and y=0
fig_dd.add_trace(
    go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        fill="tozeroy",
        name="Drawdown (%)",
        line=dict(color="#ef4444", width=1),
        fillcolor="rgba(239,68,68,0.2)",
    )
)

fig_dd.update_layout(
    title="Portfolio drawdown through time",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    template="plotly_dark",
    height=350,
    yaxis=dict(tickformat=".1f"),
)

st.plotly_chart(fig_dd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Per-asset risk breakdown
# ══════════════════════════════════════════════════════════════════════════════
st.header("7. Per-asset risk breakdown")
st.caption(
    "Understanding which positions drive your portfolio risk is core to risk management. "
    "A large weight in a low-volatility asset contributes less risk than a small weight "
    "in a high-volatility asset."
)

asset_stats = []
for ticker in valid_tickers:
    r = returns[ticker]
    vol_ann = r.std() * np.sqrt(252)    # annualise daily vol: σ_annual = σ_daily × √252
    var_hist = -np.percentile(r, (1 - confidence) * 100)
    max_dd_asset = ((1 + r).cumprod() / (1 + r).cumprod().expanding().max() - 1).min()
    asset_stats.append(
        {
            "Ticker": ticker,
            "Weight (%)": f"{raw_weights[ticker] / total_w * 100:.1f}",
            "Ann. Volatility (%)": f"{vol_ann * 100:.2f}",
            f"Hist VaR {confidence*100:.0f}% (1d, %)": f"{var_hist * 100:.2f}",
            "Max Drawdown (%)": f"{max_dd_asset * 100:.2f}",
        }
    )

asset_df = pd.DataFrame(asset_stats)
st.dataframe(asset_df, hide_index=True, use_container_width=True)

# Bar chart: annualised volatility per asset
vol_chart = go.Figure(
    go.Bar(
        x=[row["Ticker"] for row in asset_stats],
        y=[float(row["Ann. Volatility (%)"]) for row in asset_stats],
        marker_color="#3b82f6",
        text=[f"{float(row['Ann. Volatility (%)']):.1f}%" for row in asset_stats],
        textposition="outside",
    )
)
vol_chart.update_layout(
    title="Annualised volatility by asset",
    yaxis_title="Volatility (%)",
    template="plotly_dark",
    height=350,
    showlegend=False,
)
st.plotly_chart(vol_chart, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER — Interview talking points
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("💼 Interview talking points for S&T and Risk roles"):
    st.markdown(
        f"""
        Based on your current inputs, here are the numbers you could cite in an 
        interview or cover letter:

        - *"I built a risk analytics dashboard measuring {confidence*100:.0f}% 
          Historical and Parametric VaR and Expected Shortfall across a 
          {len(valid_tickers)}-asset portfolio, including a rolling backtest against 
          the Basel breach-rate standard."*

        - *"Historical VaR was £{hist_var_gbp:,.0f} vs Parametric VaR of 
          £{param_var_gbp:,.0f} on a 1-day basis — the gap reflects fat tails 
          (excess kurtosis: {kurt:.2f})."*

        - *"The ES/VaR ratio was {hist_es_pct/hist_var_pct:.2f}x, meaning the 
          average loss on a tail day is {hist_es_pct/hist_var_pct:.2f}× the VaR 
          threshold — the key figure Basel IV / FRTB focuses on."*

        - *"Maximum drawdown over the period was {max_drawdown*100:.2f}%, which I 
          used to stress-test the portfolio against a typical fund mandate 
          drawdown limit."*

        **Why ES beats VaR for regulators:** VaR is not sub-additive — the VaR of 
        a combined portfolio can *exceed* the sum of its parts, which violates basic 
        diversification logic. ES is always sub-additive, making it a coherent risk 
        measure and the correct basis for capital allocation under FRTB.
        """
    )