"""Homework 2 starter — Algory QI Education, Fall 2026

Fill in every function marked TODO. Do not rename them: the checker looks for
these exact names. Run `python check_hw02.py` before you submit.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Q1
def present_value(cash_flows, rate):
    """Present value of a list of cash flows, the first arriving in one year.

    present_value([10, 15, 20], 0.10) -> 36.51
    """
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows, start=1))


# ---------------------------------------------------------------- Q2
def bond_price(face, coupon_rate, years, market_rate):
    """Price of a bond paying an annual coupon and repaying face at maturity.

    The final year pays the coupon AND the face value. That is the usual bug.
    bond_price(1000, 0.04, 10, 0.04) -> exactly 1000.0
    """
    coupon = face * coupon_rate
    return present_value([coupon] * years, market_rate) + face / (1 + market_rate) ** years


# ---------------------------------------------------------------- Q4/Q5
def annualised_return(prices):
    """Annualised return from a price series, using 252 trading days."""
    prices = pd.Series(prices).dropna()
    return (prices.iloc[-1] / prices.iloc[0]) ** (252 / (len(prices) - 1)) - 1


def annualised_volatility(prices):
    """Annualised standard deviation of daily returns."""
    prices = pd.Series(prices).dropna()
    return prices.pct_change().dropna().std() * np.sqrt(252)


def beta(stock_prices, market_prices):
    """Beta of a stock against the market.

    Covariance of the two RETURN series divided by the variance of the market's.
    Computing this on prices instead of returns is a common and silent error.
    beta(spy, spy) -> 1.0
    """
    returns = pd.concat(
        [pd.Series(stock_prices).pct_change(), pd.Series(market_prices).pct_change()],
        axis=1,
        join="inner",
    ).dropna()
    return returns.iloc[:, 0].cov(returns.iloc[:, 1]) / returns.iloc[:, 1].var()


# ---------------------------------------------------------------- your answers
def main():
    """Everything the assignment asks you to print goes here."""
    print("Q1  present_value([10, 15, 20], 0.10) =", round(present_value([10, 15, 20], 0.10), 2))
    face, coupon_rate, years = 1000, 0.04, 10
    print("Q2  bond_price(1000, 0.04, 10, 0.04) =", round(bond_price(face, coupon_rate, years, 0.04),2))
    for market_rate in (0.02, 0.04, 0.0496):
        print(
            f"Q3  bond price at {market_rate:.2%} =",
            round(bond_price(face, coupon_rate, years, market_rate), 2),
        )

    rates = np.linspace(0.00, 0.10, 101)
    coupon = face * coupon_rate
    prices = np.empty_like(rates)
    prices[0] = coupon * years + face
    nonzero_rates = rates[1:]
    prices[1:] = coupon * (1 - (1 + nonzero_rates) ** -years) / nonzero_rates + face * (
        1 + nonzero_rates
    ) ** -years
    plt.figure()
    plt.plot(rates, prices)
    plt.xlabel("Market rate")
    plt.ylabel("Bond price")
    plt.title("Bond price vs. market rate")
    plt.savefig("q3_bond_price_vs_rate.png")
    plt.close()
    print(
        "Q3  : the convexity of the bond is slightly positive, its price increases by a larger rate when rates fall than it decreases when rates rise by the same amount."
        "its also risk protection: You lose less value when rates go up than you gain when rates go down"
    )
    tickers = ["NVDA", "XOM", "LLY"]
    symbols = tickers + ["SPY"]
    price_data = yf.download(
        symbols,
        period="5y",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    close_prices = pd.DataFrame({symbol: price_data[symbol]["Close"].dropna() for symbol in symbols})
    for symbol in symbols:
        print(f"Q4  {symbol} trading days =", close_prices[symbol].count())
    daily_returns = close_prices.pct_change().dropna()
    metrics = pd.DataFrame(index=tickers)
    metrics["return"] = (close_prices[tickers].iloc[-1] / close_prices[tickers].iloc[0]) ** (
        252 / (close_prices[tickers].count() - 1)
    ) - 1
    metrics["volatility"] = daily_returns[tickers].std() * np.sqrt(252)
    covariance = daily_returns[tickers + ["SPY"]].cov()
    metrics["beta"] = covariance.loc[tickers, "SPY"] / covariance.loc["SPY", "SPY"]
    print("Q5  return, volatility and beta")
    print(metrics)

    print("Q6 ranking tickers by beta and volatility")
    print(metrics.sort_values("beta", ascending=False)[["beta"]])
    print(metrics.sort_values("volatility", ascending=False)[["volatility"]])

if __name__ == "__main__":
    main()
