from utils import show_footer
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fredapi import Fred
import os

st.set_page_config(page_title="Macro Dashboard", layout="wide")

st.title("🌍 Macro Dashboard")
st.write(
    "Tracking the four data series that matter most to the Fed's dual mandate "
    "and to market pricing: inflation, policy rate, employment, and growth."
)

# ── Connect to FRED ─────────────────────────────────────────────
# os.environ.get() reads an environment variable.
# If it's not found, it returns None instead of crashing — 
# we check for that and show a friendly error.

api_key = os.environ.get("FRED_API_KEY") or st.secrets.get("FRED_API_KEY")

if not api_key:
    st.error(
        "FRED API key not found. Set it as an environment variable "
        "called FRED_API_KEY and restart VS Code."
    )
    st.stop()

fred = Fred(api_key=api_key)

# ── Define our series ────────────────────────────────────────────
# Each FRED series has a unique code. These are the standard ones:
# CPIAUCSL = CPI, all urban consumers, seasonally adjusted
# FEDFUNDS = Effective Federal Funds Rate
# UNRATE   = Unemployment Rate
# GDP      = Gross Domestic Product (nominal, quarterly)

SERIES = {
    "CPI (YoY % change)": "CPIAUCSL",
    "Fed Funds Rate (%)": "FEDFUNDS",
    "Unemployment Rate (%)": "UNRATE",
    "GDP (Quarterly, $bn)": "GDP",
}

# ── Download function ────────────────────────────────────────────
@st.cache_data(ttl=3600)  # macro data updates monthly/quarterly — 1hr cache is plenty
def get_fred_series(series_id: str) -> pd.Series:
    """
    Downloads a full historical series from FRED.
    Returns a pandas Series indexed by date.
    """
    return fred.get_series(series_id)

# ── Helper: classify where a value sits vs history ────────────────
def classify_level(current: float, series: pd.Series, low_pct=33, high_pct=66) -> str:
    """
    Compares the current value to its own historical distribution
    (using the last 10 years) and returns 'low', 'medium', or 'high'.
    This lets our commentary be dynamic rather than hardcoded numbers.
    """
    recent = series.tail(120)  # last ~10 years of monthly data
    low_threshold = np.percentile(recent, low_pct)
    high_threshold = np.percentile(recent, high_pct)

    if current <= low_threshold:
        return "low"
    elif current >= high_threshold:
        return "high"
    else:
        return "medium"

# ── Market commentary library ──────────────────────────────────────
# This is the core of what you asked for: a structured explanation
# of what each reading means, keyed by the classification above.
# In an interview this IS your analytical framework — you can defend
# every line here because it's standard macro reasoning.

COMMENTARY = {
    "CPI (YoY % change)": {
        "low": (
            "Inflation is running below trend. This gives the Fed room to cut "
            "rates, which typically supports equity valuations (lower discount "
            "rates) and is bearish for the dollar. Watch for deflation risk if "
            "this persists alongside weak growth."
        ),
        "medium": (
            "Inflation is roughly in line with the Fed's long-run comfort zone. "
            "Markets are likely pricing a 'steady state' policy path — limited "
            "near-term rate moves either direction. Equity multiples should be "
            "relatively stable absent other shocks."
        ),
        "high": (
            "Inflation is elevated. This pressures the Fed toward holding or "
            "hiking rates, which raises the risk-free rate used in DCF discounting "
            "— compressing equity valuations, especially for long-duration growth "
            "stocks. Typically supports the dollar and pressures bond prices "
            "(yields up, prices down)."
        ),
    },
    "Fed Funds Rate (%)": {
        "low": (
            "Policy rates are low by historical standards — an accommodative "
            "stance, usually associated with weak growth or a deliberate easing "
            "cycle. Cheap borrowing typically supports risk assets and growth "
            "equities; bond yields tend to be low."
        ),
        "medium": (
            "Policy rates are at a 'neutral' level — neither strongly stimulating "
            "nor restricting the economy. Market direction will likely be driven "
            "by incoming data (CPI, employment) rather than the rate level itself."
        ),
        "high": (
            "Policy rates are restrictive. Borrowing costs are elevated, which "
            "slows credit growth and consumer spending. This is typically bearish "
            "for equities (especially rate-sensitive sectors like real estate and "
            "tech) and supportive of the currency via capital inflows seeking yield."
        ),
    },
    "Unemployment Rate (%)": {
        "low": (
            "The labour market is tight. This can fuel wage growth and, in turn, "
            "inflation — the Fed watches this closely as part of its dual mandate. "
            "A very tight labour market alongside high inflation increases the "
            "odds of further tightening."
        ),
        "medium": (
            "Unemployment is around its historical average — a 'balanced' labour "
            "market. Limited signal on its own; pair with wage growth and job "
            "openings data for a fuller picture."
        ),
        "high": (
            "Unemployment is elevated. This signals slowing economic activity "
            "and typically pushes the Fed toward rate cuts to stimulate growth. "
            "Often coincides with weaker consumer spending and corporate earnings "
            "downgrades — bearish for cyclical equities."
        ),
    },
    "GDP (Quarterly, $bn)": {
        "low": (
            "GDP growth is weak relative to its recent trend — consistent with "
            "a slowing or contracting economy. Markets may price recession risk, "
            "which typically rotates capital toward defensive sectors and "
            "government bonds."
        ),
        "medium": (
            "GDP growth is broadly in line with trend — a 'steady state' economy. "
            "No strong directional signal on its own."
        ),
        "high": (
            "GDP growth is strong relative to trend. This is generally supportive "
            "of corporate earnings and cyclical equities, but if it coincides with "
            "tight labour markets and high inflation, it raises the odds of further "
            "Fed tightening — a classic 'good news is bad news' market dynamic."
        ),
    },
}

# ── Display each series ────────────────────────────────────────────
for name, series_id in SERIES.items():

    st.divider()
    st.subheader(name)

    raw_series = get_fred_series(series_id)
    raw_series = raw_series.dropna()

    # For CPI we convert the index level to a year-on-year % change,
    # since the raw index number (e.g. 312.3) means nothing on its own.
    if series_id == "CPIAUCSL":
        display_series = raw_series.pct_change(12) * 100  # 12 months = YoY
        display_series = display_series.dropna()
    else:
        display_series = raw_series

    current_value = display_series.iloc[-1]
    latest_date = display_series.index[-1].strftime("%B %Y")

    col_chart, col_metric = st.columns([3, 1])

    with col_metric:
        st.metric(f"Latest ({latest_date})", f"{current_value:,.2f}")

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display_series.tail(60).index,  # last 5 years of monthly data
            y=display_series.tail(60).values,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── The button you asked for ──────────────────────────────────
    # st.toggle() creates a clean on/off switch.
    # We use it instead of st.button() because we want the explanation
    # to STAY visible once opened, not disappear on the next interaction.
    show_explanation = st.toggle(
        f"What does this mean for markets?",
        key=f"toggle_{series_id}",  # unique key so each toggle works independently
    )

    if show_explanation:
        level = classify_level(current_value, display_series)
        explanation = COMMENTARY[name][level]

        # Colour-code the info box by classification
        if level == "high":
            st.warning(f"**Current reading: {level.upper()}** vs 10-year history\n\n{explanation}")
        elif level == "low":
            st.info(f"**Current reading: {level.upper()}** vs 10-year history\n\n{explanation}")
        else:
            st.success(f"**Current reading: {level.upper()}** vs 10-year history\n\n{explanation}")

# ── Combined macro regime summary ──────────────────────────────────
st.divider()
st.header("Macro Regime Summary")
st.caption("Reasoning across all four indicators together.")

# Re-fetch classifications for the summary
# (we recompute rather than storing from the loop above, for clarity)
cpi_series = get_fred_series("CPIAUCSL").pct_change(12).dropna() * 100
fed_series = get_fred_series("FEDFUNDS").dropna()
unemp_series = get_fred_series("UNRATE").dropna()
gdp_series = get_fred_series("GDP").dropna()

cpi_level = classify_level(cpi_series.iloc[-1], cpi_series)
fed_level = classify_level(fed_series.iloc[-1], fed_series)
unemp_level = classify_level(unemp_series.iloc[-1], unemp_series)
gdp_level = classify_level(gdp_series.iloc[-1], gdp_series)

# Simple rule-based regime classifier
if cpi_level == "high" and unemp_level == "low":
    regime = (
        "**Overheating** — high inflation with a tight labour market. "
        "Classic conditions for continued Fed tightening. Risk assets typically "
        "face headwinds from rising discount rates."
    )
elif cpi_level == "low" and unemp_level == "high":
    regime = (
        "**Slowdown/Recession risk** — low inflation alongside rising unemployment. "
        "The Fed has room to cut rates aggressively. Markets often rotate toward "
        "defensive sectors and government bonds."
    )
elif cpi_level == "medium" and gdp_level == "high":
    regime = (
        "**Goldilocks** — moderate inflation with strong growth. Generally the "
        "most supportive regime for risk assets, provided inflation doesn't "
        "re-accelerate."
    )
else:
    regime = (
        "**Mixed signals** — indicators are not pointing in a single clear "
        "direction. This is the most common real-world state and usually means "
        "markets will be highly sensitive to the next data print."
    )

st.markdown(regime)

st.caption(
    f"CPI: {cpi_level} · Fed Funds: {fed_level} · Unemployment: {unemp_level} · GDP: {gdp_level}"
)

show_footer()