import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class GeometricBrownianMotion:

    # Used to annualize time-scaled quantities
    TRADING_DAYS = 252

    def __init__(self, ticker, period="3y", interval="1d", auto_adjust=True):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.auto_adjust = auto_adjust

        self.price_data = self._load_prices()
        self.log_returns = self._calculate_log_returns()
        self.drift, self.volatility = self._estimate_parameters()

    # Load historical closing prices
    def _load_prices(self):
        return yf.download(self.ticker, period=self.period, interval=self.interval, auto_adjust=self.auto_adjust, progress=False)["Close"].values

    # Use historical data to calculate daily log returns
    def _calculate_log_returns(self):
        return np.log(self.price_data[1:] / self.price_data[:-1])
    
    # Estimate GBM parameters based on historical returns
    def _estimate_parameters(self):
        # Drift (μ)
        mu_daily = float(np.mean(self.log_returns) / self.TRADING_DAYS)

        # Volatility (σ)
        sigma_daily = float(np.std(self.log_returns) / np.sqrt(self.TRADING_DAYS))

        return mu_daily, sigma_daily
    
    def simulate_paths(self, n_sims=1000, n_steps=252, s0=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if s0 is None:
            s0 = float(self.price_data[-1])

        dt = 1 / self.TRADING_DAYS

        mu = self.drift
        sigma = self.volatility

        # Annualize drift and volatility
        mu_ann = mu * 252
        sigma_ann = sigma * np.sqrt(252)

        z = np.random.normal(size=(n_steps, n_sims))
        increments = (mu_ann - 0.5 * sigma_ann**2) * dt + sigma_ann * np.sqrt(dt) * z
        log_paths = np.vstack([np.zeros(n_sims), np.cumsum(increments, axis=0)])
        paths = s0 * np.exp(log_paths)
        return paths
    
    def plot_paths(self, paths, n_plot=10):
        plt.figure()
        plt.plot(paths[:, :n_plot])
        plt.title(f"GBM simulated paths: {self.ticker}")
        plt.xlabel("Steps (days)")
        plt.ylabel("Price")
        plt.show()


gbm = GeometricBrownianMotion("AAPL")
paths = gbm.simulate_paths()
gbm.plot_paths(paths)