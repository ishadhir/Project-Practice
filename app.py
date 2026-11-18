import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

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

tab1, tab2, tab3, tab4, tab_pred, tab5 = st.tabs(["Home", "Health", "Risk", "Valuation", "Prediction", "Recommendation"])


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

# ---------- PREDICTION (P/E + ROA -> next-year alpha) ----------
with tab_pred:
    st.header("Next-Year Return Prediction")
    st.write("Uses last year's P/E and ROA to predict next year's alpha "
             "(stock return minus S&P 500 return).")

    PEERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO",
             "IBM", "INTC", "ADBE", "CRM", "JNJ", "PFE", "MRK", "ABT", "KO",
             "PEP", "PG", "WMT", "COST", "HD", "MCD", "NKE", "JPM", "BAC",
             "WFC", "XOM", "CVX", "CAT"]

    @st.cache_data(ttl=86400)
    def get_prices(sym):
        px = yf.Ticker(sym).history(period="6y")["Close"]
        px.index = px.index.tz_localize(None)
        return px

    def pick_row(df, names):
        for n in names:
            if n in df.index:
                return df.loc[n]
        return None

    @st.cache_data(ttl=86400, show_spinner="Downloading data for peer stocks...")
    def build_dataset(symbols):
        mkt = get_prices("SPY")
        rows = []
        for s in symbols:
            try:
                t = yf.Ticker(s)
                inc, bal = t.income_stmt, t.balance_sheet
                ni = pick_row(inc, ["Net Income"])
                eps = pick_row(inc, ["Diluted EPS", "Basic EPS"])
                ta = pick_row(bal, ["Total Assets"])
                px = get_prices(s)
            except Exception:
                continue
            if ni is None or eps is None or ta is None:
                continue
            for d in inc.columns:
                d = pd.Timestamp(d).tz_localize(None)
                p, e, n, a = px.asof(d), eps.get(d), ni.get(d), ta.get(d)
                if pd.isna(p) or pd.isna(e) or pd.isna(n) or pd.isna(a) or e <= 0:
                    continue
                end = d + pd.DateOffset(years=1)
                if end <= px.index[-1]:   # next year's return is known
                    nxt = (px.asof(end) / p - 1) - (mkt.asof(end) / mkt.asof(d) - 1)
                else:                     # most recent year -> what we predict
                    nxt = np.nan
                rows.append({"ticker": s, "date": d, "PE": p / e,
                             "ROA": n / a, "alpha_next": nxt})
        return pd.DataFrame(rows)

    data = build_dataset(tuple(sorted(set(PEERS + [ticker]))))

    train = data.dropna(subset=["alpha_next"])
    train = train[(train["PE"] > 0) & (train["PE"] < 100)]

    X = sm.add_constant(train[["PE", "ROA"]])
    ols = sm.OLS(train["alpha_next"], X).fit()
    logit = sm.Logit((train["alpha_next"] > 0).astype(int), X).fit(disp=0)

    latest = data[data["ticker"] == ticker].sort_values("date").tail(1)
    if latest.empty:
        pred_alpha, prob_up = 0.0, 0.5
        st.warning("No positive earnings for this ticker, so no prediction.")
    else:
        x_now = pd.DataFrame({"const": [1.0],
                              "PE": [latest["PE"].iloc[0]],
                              "ROA": [latest["ROA"].iloc[0]]})
        pred_alpha = float(ols.predict(x_now)[0])
        prob_up = float(logit.predict(x_now)[0])

        c1, c2 = st.columns(2)
        c1.metric("Last year's P/E", f"{latest['PE'].iloc[0]:.1f}")
        c2.metric("Last year's ROA", f"{latest['ROA'].iloc[0]:.1%}")
        c1.metric("Predicted next-year alpha", f"{pred_alpha:+.1%}")
        c2.metric("Chance of beating S&P 500", f"{prob_up:.0%}")

    b = ols.params
    st.write(f"**Alpha(next yr) = {b['const']:.4f} + {b['PE']:.5f}·P/E + {b['ROA']:.4f}·ROA**")
    st.write(f"Observations: {int(ols.nobs)} | R²: {ols.rsquared:.3f}")
    st.dataframe(pd.DataFrame({"Coefficient": ols.params, "p-value": ols.pvalues}).round(4))

# ---------- RECOMMENDATION ----------
with tab5:
    st.header("Recommendation")
    points = 0
    if health >= 3: points += 1
    if value_per_share > price: points += 1
    if alpha > 0: points += 1
    if pred_alpha > 0: points += 1
    

    if points >= 3:
        st.success("BUY")
    elif points == 2:
        st.warning("HOLD")
    else:
        st.error("SELL")

    st.write(f"Points: {points} / 3 (health, valuation, alpha)")
    st.caption("Class project only. Not investment advice.")
