import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="portfolio Analytics", layout="wide")

st.title("📈 Portfolio Analytics")
st.write("Enter a portfolio of stocks to analyse their risk, return and correlation.")

# ── User inputs ───────────────────────────────────────────────
# st.text_input() creates a text box.
# The user types tickers separated by commas e.g. "AAPL, MSFT, GOOGL"
# We provide a default so the page isn't blank on first load.

ticker_input = st.text_input(
    "Enter Tickers (comma-separated):",
    value="AAPL,MSFT,GOOGL,AMZN,JPM"
)

# .split(",") breaks the string into a list at every comma
# .strip() removes accidental spaces around each ticker
# .upper() converts to uppercase — yfinance requires uppercase tickers

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Date range picker — two date inputs side by side
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start date", value=pd.Timestamp("2022-01-01"))

with col2:
    end_date = st.date_input("End date", value=pd.Timestamp("today"))

# Risk-free rate input — used for Sharpe ratio calculation
# We let the user adjust it so they can discuss the assumption in interviews
risk_free_rate = st.number_input(
    "Risk-free rate (%):",
    value=4.5,
    step=0.1,
    help="Typically the 3-month T-bill rate. Used to calculate Sharpe ratio."
) / 100 # decimal

# ── Download price data ───────────────────────────────────────
# st.button() returns True when clicked, False otherwise.
# We wrap everything in this if-block so nothing runs until
# the user is ready — avoids downloading on every keystroke.

if st.button("Analyse Portfolio", type="primary"):

    with st.spinner("Downloading data..."):
        @st.cache_data(ttl=300) # load_prices will only redownload after 5m (slicker but still semi live data)
        def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
            """
            Downloads adjusted closing prices for a list of tickers.
            Returns a DataFrame where each column is one ticker.
            Note: tickers must be a tuple (not list) for caching to work —
            lists can't be hashed by Streamlit's cache system.
            """
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,   # adjusts for splits and dividends
                progress=False,     # suppresses the download progress bar
            )
            # yf.download returns a multi-level DataFrame when given multiple tickers
            # ["Close"] selects just the closing prices
            return raw["Close"]
    
    prices = load_prices(
        tuple(tickers),
        str(start_date),
        str(end_date),
    )
    # Drop any columns where all values are NaN
    # (happens if the user types an invalid ticker)
    prices = prices.dropna(axis=1,how="all")

    if prices.empty:
        st.error("No data found. Check your tickers and date range")
        st.stop() # halts execution of the rest of the page

# ── Calculate daily returns ───────────────────────────────
    # .pct_change() computes (today - yesterday) / yesterday for every row
    # .dropna() removes the first row which will always be NaN
    # (there's no "previous day" for the very first data point)

    returns = prices.pct_change().dropna()

 # ── Summary statistics ────────────────────────────────────
    # We calculate these per asset (per column), then display them in a table.

    trading_days = 252

 # Mean daily return × 252 = annualised return
    ann_return = returns.mean()*trading_days

    # Daily vol × √252 = annualised volatility
    # We use square root because volatility scales with the square root of time
    # — this comes from the statistical properties of random walks,
    # which is how stock prices are modelled

    ann_vol = returns.std() * np.sqrt(trading_days)

# Sharpe ratio: excess return divided by volatility
    sharpe = (ann_return - risk_free_rate) / ann_vol

# Bundle into a clean DataFrame for display
    summary = pd.DataFrame({
        "Annualised Return (%)": (ann_return * 100).round(2),
        "Annualised Volatility (%)": (ann_vol * 100).round(2),
        "Sharpe Ratio": sharpe.round(3),
    })

    # ── Display summary table ─────────────────────────────────
    st.subheader("Risk & Return Summary")
    # absolute colour coding to compare returns visually, using chains

    styled = summary.style \
    .background_gradient(
        cmap="RdYlGn",
        subset=["Annualised Return (%)"],
        vmin=-20,
        vmax=20
    ) \
    .background_gradient(
        cmap="RdYlGn_r",
        subset=["Annualised Volatility (%)"],
        vmin=10,
        vmax=40
    ) \
    .background_gradient(
        cmap="RdYlGn",
        subset=["Sharpe Ratio"],
        vmin=-0.5,
        vmax=2.0
    ) \
    .format({
        "Annualised Return (%)": "{:.2f}",
        "Annualised Volatility (%)": "{:.2f}",
        "Sharpe Ratio": "{:.3f}"
    }) \
    .set_properties(**{
        "font-size": "14px",
        "text-align": "center",
    }) \
    .set_table_styles([{
        "selector": "th",
        "props": [("font-size", "13px"), ("text-align", "center")]
    }])

    # — green for high values, red for low, like a heat map in Excel
    st.dataframe(
        styled, use_container_width=True
    )

    # ── Cumulative returns chart ──────────────────────────────
    # We want to show how £100 invested in each stock would have grown.
    # Formula: (1 + daily_return).cumprod() gives the growth multiple.
    # Multiply by 100 to start everything at £100.

    cumulative = (1 + returns).cumprod() * 100
    st.subheader("Cumulative Return (£100 Invested)")
    fig_cum = go.Figure()

    for col in cumulative.columns:
        fig_cum.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col],
            mode="lines",
            name=col,
        ))

    fig_cum.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value (£)",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom",y=1.02),
    )
    st.plotly_chart(fig_cum,use_container_width=True)

# ── Correlation matrix ────────────────────────────────────
    # .corr() computes pairwise correlations between all columns
    # Returns a square matrix: each cell is the correlation
    # between that row's asset and that column's asset

    corr = returns.corr().round(2)
    # Create a mask — np.triu returns a matrix of True/False
    # True where we want to HIDE values (upper triangle + diagonal... 
    # or keep diagonal, your choice)
    # np.ones_like creates a matrix of 1s the same shape as corr
    # np.tril then keeps only the lower triangle as True, rest False
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask,k=1)] = True
    # k=1 means mask above the diagonal — diagonal itself stays visible (k=0 to remove diagonal entries)

    # Replace masked values with NaN so imshow leaves them blank
    corr_masked = corr.mask(mask)

    st.subheader("Correlation Matrix")
    st.write("Values closer to +1 mean assets move together. Closer to 0 means more diversification benefit.")

    # px.imshow() turns a DataFrame into a heatmap
    # color_continuous_scale="RdYlGn" = red (low) → yellow → green (high)
    # zmin/zmax fix the colour scale from -1 to +1

    fig_corr = px.imshow(
        corr_masked,
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        text_auto=True   # displays the number inside each cell
    )

    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

st.divider()
st.caption("© 2026 Toby Medley. Shared for demonstration purposes only. All rights reserved.")