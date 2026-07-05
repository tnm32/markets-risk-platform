import streamlit as st
import yfinance as yf

st.title("Markets & Risk Analytics Platform")
st.header("📊 Market Overview")

# ── Define the assets we want to track ────────────────────────
# This is a Python dictionary: each key is the display name,
# each value is the Yahoo Finance ticker symbol.
# Indices use ^ prefix; FX pairs use =X suffix; bonds use ^TNX.
ASSETS = {
    "S&P 500":       "^GSPC",
    "Nasdaq":        "^IXIC",
    "FTSE 100":      "^FTSE",
    "DAX":           "^GDAXI",
    "GBP/USD":       "GBPUSD=X",
    "EUR/USD":       "EURUSD=X",
    "US 10Y Yield":  "^TNX",
}

# ── Download price data for all assets ────────────────────────
# We fetch 5 days of data for each asset.
# We use a cache decorator so Streamlit doesn't re-download
# every single time you interact with the page — it stores
# the result for 5 minutes (300 seconds).

@st.cache_data(ttl=300)
def get_market_data(tickers: list[str]) -> dict:
    """
    Downloads the last 5 days of closing prices for each ticker.
    Returns a dict of { ticker: DataFrame }.
    """
    data = {}
    for ticker in tickers:
        obj = yf.Ticker(ticker)
        data[ticker] = obj.history(period="5d")
    return data

# Grab all tickers from our ASSETS dictionary
all_tickers = list(ASSETS.values())
market_data = get_market_data(all_tickers)

# ── Helper function: extract latest price and daily change ─────
# A function is a reusable block of code. We define it once
# and call it for each of our 7 assets.

def get_price_and_delta(df):
    """
    Given a DataFrame of OHLCV data, returns:
    - latest closing price
    - absolute daily change (today vs yesterday)
    - percentage daily change
    """
    if df.empty or len(df) < 2:
        return None, None, None

    latest = df["Close"].iloc[-1]       # most recent close
    previous = df["Close"].iloc[-2]     # one day before that
    change = latest - previous          # absolute change
    pct_change = (change / previous) * 100   # percentage change

    return latest, change, pct_change

# ── Display assets in a 3-column grid ─────────────────────────
# st.columns(3) creates 3 side-by-side columns.
# We use a loop so we don't have to write the same code 7 times.

cols = st.columns(3)   # creates a list of 3 column objects

for i, (name, ticker) in enumerate(ASSETS.items()):
    # enumerate() gives us both the index (i) and the key-value pair
    # i % 3 cycles through 0, 1, 2, 0, 1, 2 — placing each asset
    # in the correct column
    col = cols[i % 3]

    df = market_data[ticker]
    price, change, pct = get_price_and_delta(df)
    if price is None:
        col.warning(f"No data for {name}")
        continue

    # Format the delta string for st.metric
    # For FX and yield we show more decimal places
    if ticker in ("GBPUSD=X", "EURUSD=X"):
        price_str = f"{price:.4f}"
        delta_str = f"{change:+.4f} ({pct:+.2f}%)"
    elif ticker == "^TNX":
        price_str = f"{price:.3f}%"
        delta_str = f"{change:+.3f}pp ({pct:+.2f}%)"
    else:
        price_str = f"{price:,.2f}"
        delta_str = f"{change:+.2f} ({pct:+.2f}%)"

    # st.metric(label, value, delta)
    # Streamlit automatically colours delta green if positive,
    # red if negative — just like a Bloomberg terminal
    col.metric(
        label=name,
        value=price_str,
        delta=delta_str,
    )

# ── Divider and mini chart section ────────────────────────────
st.divider()
st.subheader("Price History")

# st.selectbox() creates a dropdown menu
selected_name = st.selectbox(
    "Select an asset to chart:",
    options=list(ASSETS.keys())
)

selected_ticker = ASSETS[selected_name]
# For the chart we pull 3 months of data — more informative
@st.cache_data(ttl=300)
def get_chart_data(ticker:str):
    obj = yf.Ticker(ticker)
    return obj.history(period="3mo")

chart_df = get_chart_data(selected_ticker)

if not chart_df.empty:
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df["Close"],
        mode="lines",
        name=selected_name,
        line=dict(color="#1f77b4", width=2),
    ))

    fig.update_layout(
        title=f"{selected_name} — Last 3 Months",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",   # shows all values at a given x on hover
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Chart data unavailable.")