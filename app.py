import streamlit as st
import yfinance as yf
import numpy as np

st.title("Stock Health Check")

ticker = st.text_input("Enter a ticker", "AAPL").upper()
stock = yf.Ticker(ticker)
info = stock.info
price = stock.history(period="5d")["Close"].iloc[-1]
income = stock.financials
balance = stock.balance_sheet
cashflow = stock.cashflow

def get(table, row):
    try:
        return float(table.loc[row].dropna().iloc[0])
    except Exception:
        return None

def ratio(a, b):
    return a / b if a and b else None

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Health", "Risk", "Valuation", "Recommendation"])

# ---------- HOME ----------
with tab1:
    st.header(info.get("longName", ticker))
    st.write("Sector:", info.get("sector"))
    st.metric("Current Price", f"${price:,.2f}")

# ---------- HEALTH ----------
with tab2:
    st.header("Financial Health")
    current_ratio = ratio(get(balance, "Current Assets"), get(balance, "Current Liabilities"))
    debt_equity = ratio(get(balance, "Total Debt"), get(balance, "Stockholders Equity"))
    profit_margin = ratio(get(income, "Net Income"), get(income, "Total Revenue"))
    roa = ratio(get(income, "Net Income"), get(balance, "Total Assets"))

    st.metric("Current Ratio", round(current_ratio, 2) if current_ratio else "N/A")
    st.metric("Debt / Equity", round(debt_equity, 2) if debt_equity else "N/A")
    st.metric("Profit Margin", f"{profit_margin:.1%}" if profit_margin else "N/A")
    st.metric("Return on Assets", f"{roa:.1%}" if roa else "N/A")

    health = 0
    if current_ratio and current_ratio > 1.5: health += 1
    if debt_equity and debt_equity < 1.0: health += 1
    if profit_margin and profit_margin > 0.10: health += 1
    if roa and roa > 0.05: health += 1
    st.subheader(f"Health Score: {health} / 4")


# ---------- RISK (CAPM) ----------
with tab3:
    st.header("Risk: CAPM Beta")
    prices = yf.download([ticker, "SPY"], period="5y", interval="1mo", auto_adjust=True)["Close"]
    returns = prices.pct_change().dropna()

    beta, alpha = np.polyfit(returns["SPY"], returns[ticker], 1)

    st.metric("Beta", round(beta, 2))
    st.metric("Alpha (monthly)", f"{alpha:.2%}")
    st.scatter_chart(returns, x="SPY", y=ticker)

# ---------- VALUATION (DCF) ----------
with tab4:
    st.header("DCF Valuation")
    r = st.slider("Discount rate (r)", 0.06, 0.15, 0.09)
    g = st.slider("Growth rate (g)", 0.00, 0.05, 0.03)

    fcf = get(cashflow, "Free Cash Flow")
    shares = get(balance, "Ordinary Shares Number")

    if fcf and fcf > 0:
        value_per_share = (fcf * (1 + g) / (r - g)) / shares
        st.metric("Intrinsic Value per Share", f"${value_per_share:,.2f}")
    else:
        value_per_share = 0
        st.warning("Free cash flow is negative or missing, so DCF can't be used.")

    st.metric("Market Price", f"${price:,.2f}")

# ---------- RECOMMENDATION ----------
with tab5:
    st.header("Recommendation")
    points = 0
    if health >= 3: points += 1
    if value_per_share > price: points += 1
    if alpha > 0: points += 1

    if points >= 2:
        st.success("BUY")
    elif points == 1:
        st.warning("HOLD")
    else:
        st.error("SELL")

    st.write(f"Points: {points} / 3 (health, valuation, alpha)")
    st.caption("Class project only. Not investment advice.")
