from utils import show_footer
import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Options Analytics", layout="wide")

st.title("📐 Options Analytics")
st.write(
    "Black-Scholes option pricing and the Greeks — the core toolkit "
    "used on any derivatives or volatility desk."
)

with st.expander("📚 Finance concepts"):
    st.markdown(
        """
        **Black-Scholes** prices a European option using five inputs: spot price, 
        strike, time to expiry, volatility, and the risk-free rate.

        **The Greeks** measure sensitivity:
        - **Delta** — price change per £1 move in the underlying (≈ probability ITM)
        - **Gamma** — how fast Delta itself changes (curvature)
        - **Vega** — price change per 1% move in volatility

        **Key intuition**: options are fundamentally bets on volatility. Two options 
        with identical strikes and expiries but different implied volatilities 
        will have very different prices, even if the underlying doesn't move.
        """
    )

# ── Black-Scholes pricing functions ────────────────────────────
# These implement the closed-form solution directly from the
# Black-Scholes-Merton (1973) formula — not a library shortcut.

def calculate_d1_d2(S, K, T, r, sigma):
    """
    d1 and d2 are intermediate terms used throughout the Black-Scholes
    formula. They essentially measure 'how many standard deviations
    away from the strike' the current price is, adjusted for drift.

    d1 = [ln(S/K) + (r + 0.5σ²)T] / (σ√T)
    d2 = d1 - σ√T
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call(S, K, T, r, sigma):
    """
    Call option price.
    Formula: C = S × N(d1) − K × e^(−rT) × N(d2)

    Intuition:
    - S × N(d1) is the expected benefit from owning the stock if exercised
    - K × e^(−rT) × N(d2) is the present value of paying the strike,
      weighted by the probability of exercise
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """
    Put option price, using put-call parity logic directly:
    P = K × e^(−rT) × N(−d2) − S × N(−d1)
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# ── Greeks ────────────────────────────────────────────────────
# Each Greek is a partial derivative of the option price with
# respect to one input. These have closed-form solutions too.

def calculate_delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta: ∂(option price) / ∂S
    Call delta = N(d1)        — ranges from 0 to 1
    Put delta  = N(d1) − 1    — ranges from -1 to 0
    """
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def calculate_gamma(S, K, T, r, sigma):
    """
    Gamma: ∂²(option price) / ∂S²
    Same formula for calls and puts — Gamma is identical for both
    at the same strike and expiry.

    Formula: φ(d1) / (S × σ × √T)
    where φ is the standard normal PROBABILITY DENSITY function
    (different from the cumulative one used above).
    """
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calculate_vega(S, K, T, r, sigma):
    """
    Vega: ∂(option price) / ∂σ
    Same formula for calls and puts.
    Formula: S × φ(d1) × √T
    Conventionally divided by 100 so it represents the price change
    per 1 PERCENTAGE POINT move in volatility (e.g. 20% → 21%),
    rather than per 1.0 (i.e. 100 percentage points).
    """
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100

st.header("1. Option parameters")

col1, col2, col3 = st.columns(3)

with col1:
    S = st.number_input(
        "Spot price (£)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        help="The current market price of the underlying asset."
    )
    K = st.number_input(
        "Strike price (£)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        help="The fixed price at which the option holder can buy (call) or sell (put) the underlying. Set equal to spot for an 'at-the-money' option."
    )

with col2:
    T_days = st.number_input(
        "Time to expiry (days)",
        min_value=1,
        value=90,
        step=1,
        help="Calendar days until the option expires. Converted to years (÷365) for the Black-Scholes formula, which requires time in annualised terms."
    )
    T = T_days / 365

with col3:
    sigma = st.number_input(
        "Volatility (annualised, %)",
        min_value=0.1,
        value=25.0,
        step=0.5,
        help="Annualised standard deviation of the underlying's returns. The only input not directly observable in the market — usually estimated from historical price moves or implied from other option prices. Higher volatility increases option value regardless of direction."
    ) / 100
    r = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        value=4.5,
        step=0.1,
        help="Typically the short-term government bond yield. Used to discount the strike price back to present value in the Black-Scholes formula."
    ) / 100

st.header("2. Pricing & Greeks")

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

call_delta = calculate_delta(S, K, T, r, sigma, "call")
put_delta = calculate_delta(S, K, T, r, sigma, "put")
gamma = calculate_gamma(S, K, T, r, sigma)
vega = calculate_vega(S, K, T, r, sigma)

col_call, col_put = st.columns(2)

with col_call:
    st.subheader("📈 Call Option")
    st.metric("Price", f"£{call_price:.2f}")
    st.metric("Delta", f"{call_delta:.3f}")
    st.metric("Gamma", f"{gamma:.4f}")
    st.metric("Vega", f"{vega:.3f}")

with col_put:
    st.subheader("📉 Put Option")
    st.metric("Price", f"£{put_price:.2f}")
    st.metric("Delta", f"{put_delta:.3f}")
    st.metric("Gamma", f"{gamma:.4f}")
    st.metric("Vega", f"{vega:.3f}")

st.caption(
    "Gamma and Vega are identical for calls and puts at the same strike "
    "and expiry — a property that follows directly from put-call parity."
)

st.header("3. Price sensitivity to the underlying")

# Generate a range of spot prices around the current one
spot_range = np.linspace(S * 0.5, S * 1.5, 100)

call_prices = [black_scholes_call(s, K, T, r, sigma) for s in spot_range]
put_prices = [black_scholes_put(s, K, T, r, sigma) for s in spot_range]

fig_sens = go.Figure()

fig_sens.add_trace(go.Scatter(
    x=spot_range, y=call_prices, mode="lines", name="Call price",
    line=dict(color="#22c55e", width=2)
))
fig_sens.add_trace(go.Scatter(
    x=spot_range, y=put_prices, mode="lines", name="Put price",
    line=dict(color="#ef4444", width=2)
))

# Mark the current spot price with a vertical line
fig_sens.add_vline(
    x=S, line=dict(color="grey", dash="dash"),
    annotation_text="Current spot"
)

fig_sens.update_layout(
    xaxis_title="Underlying price (£)",
    yaxis_title="Option price (£)",
    height=400,
    hovermode="x unified",
)

st.plotly_chart(fig_sens, use_container_width=True)


st.header("4. Payoff diagram at expiry")

option_type_payoff = st.radio("Show payoff for:", ["Call", "Put"], horizontal=True)

if option_type_payoff == "Call":
    premium = call_price
    payoff = np.maximum(spot_range - K, 0) - premium
else:
    premium = put_price
    payoff = np.maximum(K - spot_range, 0) - premium

fig_payoff = go.Figure()

fig_payoff.add_trace(go.Scatter(
    x=spot_range, y=payoff, mode="lines",
    line=dict(color="#3b82f6", width=2),
    name="P&L at expiry",
))

# Zero line — breakeven reference
fig_payoff.add_hline(y=0, line=dict(color="grey", dash="dot"))
fig_payoff.add_vline(x=K, line=dict(color="grey", dash="dash"), annotation_text="Strike")

fig_payoff.update_layout(
    xaxis_title="Underlying price at expiry (£)",
    yaxis_title="Profit / Loss (£)",
    height=400,
)

st.plotly_chart(fig_payoff, use_container_width=True)