import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go

# Page Config

st.title("Portfolio Optimisation")

st.write(
    "Construct optimal portfolios using Modern Portfolio Theory (Markowitz, 1952). "
    "Find the Maximum Sharpe and Minimum Variance portfolios, and plot the Efficient Frontier."
)


# user input:

col1, col2, col3 = st.columns(3)
 
with col1:
    tickers = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,NVDA,JPM")
 
with col2:
    period = st.selectbox("Lookback period", ["1y", "2y", "3y", "5y"], index=2)
 
with col3:
    # Risk-free rate: in interviews always justify this.
    # ~5% reflects recent US 3-month T-bill yield; adjust as markets change.
    rf_rate = st.number_input(
        "Risk-free rate (annual, %)",
        min_value=0.0,
        max_value=20.0,
        value=4.5,
        step=0.1
    ) / 100  # convert to decimal
 
ticker_list = [t.strip().upper() for t in tickers.split(",")]
 
# Guard: optimisation is meaningless with only one asset
if len(ticker_list) < 2:
    st.error("Please enter at least 2 tickers for portfolio optimisation.")
    st.stop()

# Download price data:

@st.cache_data
def load_data(tickers: list, period: str) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    @st.cache_data stores the result — Streamlit only re-runs this function
    when tickers or period actually change, not on every page interaction.
    """
    data = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    # If only one ticker is returned, yfinance gives a Series; force DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna()
 
data = load_data(ticker_list, period)

# Check all tickers downloaded
missing = [t for t in ticker_list if t not in data.columns]
if missing: 
    st.warning(f"Could not find data for: {' ,'.join(missing)}, Proceeding with available tickers.")
    ticker_list = [t for t in ticker_list if t in data.columns]
    data = data[ticker_list]

if len(ticker_list) < 2:
    st.error("Not enough valid tickers. Please revise your input.")
    st.stop

# Calc & display returns - use returns for stock comparable results 

returns = data.pct_change().dropna()
 
# Annualised figures (252 trading days in a year)
annual_returns = returns.mean() * 252          # Expected return per asset
cov_matrix = returns.cov() * 252               # Annualised covariance matrix
 
with st.expander("Show raw data & statistics"):
    st.subheader("Price Data (last 5 rows)")
    st.dataframe(data.tail())
 
    st.subheader("Expected Annual Returns")
    st.dataframe(annual_returns.rename("Expected Return").map("{:.2%}".format))
 
    st.subheader("Annualised Covariance Matrix")
    st.dataframe(cov_matrix.style.format("{:.4f}"))
 

# Portfolio Functions

def portfolio_return(weights: np.ndarray) -> float:
    """Weighted sum of individual asset expected returns."""
    return float(np.dot(weights, annual_returns))
 
def portfolio_volatility(weights: np.ndarray) -> float:
    """
    Portfolio standard deviation from the covariance matrix.
    Formula: sqrt(w^T · Σ · w)
    This is the key MPT result — it captures co-movement between assets,
    so diversification automatically reduces volatility below the
    weighted average of individual asset vols.
    """
    return float(np.sqrt(weights.T @ cov_matrix.values @ weights))

def sharpe_ratio(weights: np.ndarray, rf: float) -> float:
    """(Portfolio Return - Risk-Free Rate) / Portfolio Volatility."""
    return (portfolio_return(weights) - rf) / portfolio_volatility(weights)

#  OPTIMISATION SETUP  (shared constraints & bounds)

n = len(ticker_list)
initial_weights = np.array([1/n]*n)    # Equal weight starting point

# Constraint: weights must sum to 1 (fully invested)
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

# Bounds: each weight between 0% and 100% (long-only, no short selling)
# In practice, many funds allow shorting; we keep it simple here.

bounds = tuple((0.0,1.0) for _ in range(n))

# MAX SHARPE PORTFOLIO

def neg_sharpe(weights: np.ndarray) -> float:
    """
    We MINIMISE the negative Sharpe ratio, which is equivalent to
    MAXIMISING the Sharpe ratio. scipy.optimize only minimises.
    """
    return -sharpe_ratio(weights, rf_rate)

result_sharpe = minimize(
    neg_sharpe,
    initial_weights,
    method="SLSQP",   # Sequential Least Squares Programming — standard for constrained optimisation
    bounds = bounds,
    constraints = constraints,
    options={"ftol": 1e-9, "maxiter": 1000}
)

# Extract results
max_sharpe_weights = result_sharpe.x
max_sharpe_ret     = portfolio_return(max_sharpe_weights)
max_sharpe_vol     = portfolio_volatility(max_sharpe_weights)
max_sharpe_sr      = sharpe_ratio(max_sharpe_weights, rf_rate)

#  MINIMUM VARIANCE PORTFOLIO

result_minvol = minimize(
    portfolio_volatility,          # Minimise volatility directly
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-9, "maxiter": 1000}
)

min_vol_weights = result_minvol.x
min_vol_ret     = portfolio_return(min_vol_weights)
min_vol_vol     = portfolio_volatility(min_vol_weights)
min_vol_sr      = sharpe_ratio(min_vol_weights, rf_rate)

#  EFFICIENT FRONTIER  (Monte Carlo simulation)
# We generate thousands of random portfolios to trace the frontier.
# Each point is a random set of weights (summing to 1); we compute
# return and vol for each. The upper edge of this cloud IS the frontier.

NUM_PORTFOLIOS = 3000
sim_returns = []
sim_vols    = []
sim_sharpes = []
sim_weights = []

rng = np.random.default_rng(42) # set seed
for _ in range(NUM_PORTFOLIOS):
    w = rng.random(n)   # generate n random numbers, one per asset
    w /=w.sum()         # divide each by their sum → weights now sum to 1
    r = portfolio_return(w)
    v = portfolio_volatility(w)
    sim_returns.append(r)
    sim_vols.append(v)
    sim_sharpes.append((r - rf_rate) / v)
    sim_weights.append(w)

sim_returns = np.array(sim_returns)
sim_vols    = np.array(sim_vols)
sim_sharpes = np.array(sim_sharpes)

#  EQUAL-WEIGHT BENCHMARK

ew_weights = initial_weights
ew_ret     = portfolio_return(ew_weights)
ew_vol     = portfolio_volatility(ew_weights)
ew_sr      = sharpe_ratio(ew_weights, rf_rate)

#  DISPLAY: OPTIMISED PORTFOLIO METRICS

st.divider()
st.subheader("Optimised Portfolio Results")
 
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.markdown("**⚖️ Equal Weight (Benchmark)**")
    st.metric("Expected Return", f"{ew_ret:.2%}")
    st.metric("Volatility",      f"{ew_vol:.2%}")
    st.metric("Sharpe Ratio",    f"{ew_sr:.3f}")


with metric_col2:
    st.markdown("**📈 Maximum Sharpe Portfolio**")
    st.metric("Expected Return", f"{max_sharpe_ret:.2%}", delta=f"{max_sharpe_ret - ew_ret:+.2%} vs EW")
    st.metric("Volatility",      f"{max_sharpe_vol:.2%}", delta=f"{max_sharpe_vol - ew_vol:+.2%} vs EW", delta_color="inverse")
    st.metric("Sharpe Ratio",    f"{max_sharpe_sr:.3f}",  delta=f"{max_sharpe_sr - ew_sr:+.3f} vs EW")
 

with metric_col3:
    st.markdown("**🛡️ Minimum Variance Portfolio**")
    st.metric("Expected Return", f"{min_vol_ret:.2%}",  delta=f"{min_vol_ret - ew_ret:+.2%} vs EW")
    st.metric("Volatility",      f"{min_vol_vol:.2%}",  delta=f"{min_vol_vol - ew_vol:+.2%} vs EW", delta_color="inverse")
    st.metric("Sharpe Ratio",    f"{min_vol_sr:.3f}",   delta=f"{min_vol_sr - ew_sr:+.3f} vs EW")
 

#  DISPLAY: WEIGHT TABLES

st.divider()
st.subheader("Portfolio Weights")
 
weight_col1, weight_col2, weight_col3 = st.columns(3)

def weight_df(weights):
    return pd.DataFrame({
        "Ticker": ticker_list,
        "Weight": [f"{w:.2%}" for w in weights]
    })
 
with weight_col1:
    st.markdown("**Equal Weight**")
    st.dataframe(weight_df(ew_weights), hide_index=True)
 
with weight_col2:
    st.markdown("**Max Sharpe**")
    st.dataframe(weight_df(max_sharpe_weights), hide_index=True)
 
with weight_col3:
    st.markdown("**Min Variance**")
    st.dataframe(weight_df(min_vol_weights), hide_index=True)


#  DISPLAY: EFFICIENT FRONTIER CHART

st.divider()
st.subheader("Efficient Frontier")
st.caption(
    f"Each dot is a randomly-generated portfolio ({NUM_PORTFOLIOS:,} simulated). "
    "Colour indicates Sharpe ratio. The upper-left edge of the cloud is the efficient frontier."
)


fig = go.Figure()
# Random portfolios — coloured by Sharpe ratio
fig.add_trace(go.Scatter(
    x=sim_vols,
    y=sim_returns,
    mode='markers',
    marker=dict(
        color=sim_sharpes,
        colorscale="viridis",
        size=4,
        opacity=0.6,
        colorbar=dict(title="Sharpe Ratio", thickness=14),
        showscale=True
    ),
    name="Simulated Portfolios",
    hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
))

# Individual assets — so you can see where each stock sits vs the portfolios

for i, ticker in enumerate(ticker_list):
    asset_vol=np.sqrt(cov_matrix.iloc[i,i])     # Diagonal entries [i,i] are each asset's variance (Cov of asset with itself).
    asset_ret = float(annual_returns.iloc[i])
    fig.add_trace(go.Scatter(
        x=[asset_vol],
        y=[asset_ret],
        mode="markers+text",
        marker=dict(size=12, symbol="diamond", color="white",line=dict(color="black", width=1.5)),
        text=[ticker],
        textposition="top center",
        name=ticker,
        showlegend=True
    ))

# Max Sharpe Portfolio

fig.add_trace(go.Scatter(
    x=[max_sharpe_vol],
    y=[max_sharpe_ret],
    mode="markers",
    marker=dict(size=18,symbol="star",color="gold", line=dict(color="black", width=1.5)),
    name=f"Max Sharpe ({max_sharpe_sr:.2f})",
    hovertemplate=f"Max Sharpe<br>Vol: {max_sharpe_vol:.2%}<br>Return: {max_sharpe_ret:.2%}<extra></extra>"
))

# Minimum Variance Portfolio

fig.add_trace(go.Scatter(
    x=[min_vol_vol],
    y=[min_vol_ret],
    mode="markers",
    marker=dict(size=18, symbol="pentagon", color="red", line=dict(color="black", width=1.5)),
    name=f"Min Variance",
    hovertemplate=f"Min Variance<br>Vol: {min_vol_vol:.2%}<br>Return: {min_vol_ret:.2%}<extra></extra>"
))

# Equal Weight

fig.add_trace(go.Scatter(
    x=[ew_vol],
    y=[ew_ret],
    mode="markers",
    marker=dict(size=14, symbol="circle", color="cyan", line=dict(color="black", width=1.5)),
    name="Equal Weight",
    hovertemplate=f"Equal Weight<br>Vol: {ew_vol:.2%}<br>Return: {ew_ret:.2%}<extra></extra>"
))

fig.update_layout(
    xaxis_title="Annualised Volatility (Risk)",
    yaxis_title="Annualised Expected Return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    legend=dict(orientation="v", x=1.15, y=1),
    height=550,
    margin=dict(r=160)
)

st.plotly_chart(fig, use_container_width=True)

#  INTERVIEW CALLOUTS

st.divider()
with st.expander("📋 Interview Talking Points — what this page demonstrates"):
    st.markdown("""
    **What is the efficient frontier?**
    > The set of portfolios that maximise expected return for a given level of risk (or equivalently,
    minimise risk for a given return). Any portfolio below the frontier is suboptimal — dominated
    by one that offers more return for the same risk, or less risk for the same return.
 
    **What is the Maximum Sharpe (tangency) portfolio?**
    > The portfolio on the efficient frontier with the highest risk-adjusted return. Under CAPM,
    all rational mean-variance investors hold this portfolio combined with the risk-free asset.
    The mix between them (the Capital Allocation Line) depends on individual risk tolerance.
 
    **What is the Minimum Variance portfolio?**
    > The leftmost point on the frontier — lowest achievable volatility given these assets.
    Relevant for risk-parity and capital preservation mandates.
 
    **How does diversification reduce risk here?**
    > Portfolio volatility is `√(wᵀΣw)`, not the weighted average of individual vols.
    Off-diagonal covariance terms (correlations) reduce the total — if two assets move
    partly in opposite directions, the portfolio vol falls below either individual vol.
 
    **Limitations to raise proactively:**
    - Inputs (expected returns, covariance) are estimated from historical data — *garbage in, garbage out*.
    - The model is long-only and ignores transaction costs, liquidity, and taxes.
    - Expected returns are notoriously unstable; in practice many PMs use Black-Litterman
      to blend market implied returns with their own views.
    - Optimisation tends to concentrate weights in a few assets — real portfolios add turnover and
      concentration constraints.
    """)