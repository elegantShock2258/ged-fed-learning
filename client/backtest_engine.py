"""
Backtest Engine for the Finance Hedge Fund Agent.

Replays daily ETF data (cached locally via yfinance) and tracks the agent's
cumulative portfolio return vs. a SPY buy-and-hold benchmark.

Metrics produced:
  - Cumulative portfolio return (%)
  - SPY benchmark return (%)
  - Annualized Sharpe ratio
  - Maximum drawdown
  - Information Ratio (alpha over benchmark)
  - Rolling 30-day Sharpe timeseries
"""

import os
import numpy as np
import logging
import json
from datetime import datetime

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

log = logging.getLogger(__name__)

# Sector ETF tickers (one per GICS sector)
SECTOR_TICKERS = [
    "XLK", "XLF", "XLV", "XLE", "XLB",
    "XLI", "XLU", "XLRE", "XLP", "XLY", "XLC"
]
BENCHMARK_TICKER = "SPY"


def _download_prices(tickers, period="2y"):
    """Download daily adjusted close prices. Falls back to synthetic GBM if yf unavailable."""
    if not HAS_YF:
        log.warning("yfinance not installed — using synthetic GBM for backtest.")
        n_days = 504  # ~2 years of trading days
        prices = {}
        np.random.seed(42)
        for t in tickers:
            p = 100.0
            series = [p]
            for _ in range(n_days - 1):
                p *= np.exp(np.random.normal(0.0003, 0.012))
                series.append(p)
            prices[t] = np.array(series)
        return prices, n_days

    try:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
        data = data.dropna()
        prices = {t: data[t].values for t in SECTOR_TICKERS if t in data.columns}
        spy_prices = data[BENCHMARK_TICKER].values if BENCHMARK_TICKER in data.columns else None
        return prices, spy_prices
    except Exception as e:
        log.warning(f"yfinance download failed: {e}")
        return None, None


class BacktestEngine:
    """
    Replay daily ETF data and evaluate an agent policy against SPY benchmark.

    Usage:
        engine = BacktestEngine(model, env_class=FinanceTradingEnv)
        results = engine.run()
    """
    def __init__(self, model, device, env_class=None, period="2y", output_dir="saved_models/finance"):
        self.model = model
        self.device = device
        self.env_class = env_class
        self.period = period
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _agent_decide(self, obs):
        """Run the policy to get an execution action."""
        import torch
        self.model.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    def run(self):
        """
        Run a full backtest and return summary metrics.
        Simulates daily portfolio decisions driven by the trained policy.
        """
        all_tickers = SECTOR_TICKERS + [BENCHMARK_TICKER]
        prices_raw, spy_prices = _download_prices(all_tickers, self.period)

        if prices_raw is None:
            log.error("Could not retrieve price data for backtest.")
            return {}

        # Build daily returns matrix: [n_days-1, n_sectors]
        n_sectors = len(SECTOR_TICKERS)
        min_days = min(len(v) for v in prices_raw.values())
        price_matrix = np.column_stack([prices_raw[t][:min_days] for t in SECTOR_TICKERS])
        daily_returns = np.diff(price_matrix, axis=0) / (price_matrix[:-1] + 1e-9)  # [n_days-1, n_sectors]

        spy_returns = np.diff(spy_prices[:min_days]) / (spy_prices[:min_days - 1] + 1e-9)

        # Portfolio state: equal initial weights, rebalance when agent says to
        weights = np.ones(n_sectors) / n_sectors
        portfolio_value = 1.0
        portfolio_curve = [1.0]
        spy_value = 1.0
        spy_curve = [1.0]

        env = self.env_class() if self.env_class else None

        for day in range(len(daily_returns)):
            # Daily return for current weights
            day_return = float(np.dot(weights, daily_returns[day]))
            portfolio_value *= (1 + day_return)
            portfolio_curve.append(portfolio_value)
            
            spy_value *= (1 + float(spy_returns[day]))
            spy_curve.append(spy_value)

            # Agent makes a decision every day using env
            if env is not None:
                obs = env.reset()
                done = False
                exec_action = 33  # default: rebalance
                while not done:
                    action = self._agent_decide(obs)
                    obs, _, done, info = env.step(action)
                    if done and action >= env.num_features:
                        exec_action = action

                if exec_action == 33:
                    # Rebalance: redistribute toward Sharpe-optimal weights
                    sector_rets = daily_returns[max(0, day - 20):day + 1]
                    if len(sector_rets) > 5:
                        avg_r = sector_rets.mean(axis=0)
                        std_r = sector_rets.std(axis=0) + 1e-9
                        sharpe_w = np.clip(avg_r / std_r, 0, None)
                        total_w = sharpe_w.sum()
                        weights = sharpe_w / total_w if total_w > 0 else weights
                elif exec_action == 34:
                    # Liquidate: move to cash (uniform = benchmark proxy)
                    weights = np.ones(n_sectors) / n_sectors

        # Compute summary metrics
        port_arr = np.array(portfolio_curve)
        spy_arr = np.array(spy_curve)
        port_daily = np.diff(port_arr) / port_arr[:-1]
        spy_daily = np.diff(spy_arr) / spy_arr[:-1]

        sharpe = self._annualized_sharpe(port_daily)
        max_dd = self._max_drawdown(port_arr)
        total_return = float((port_arr[-1] - 1.0) * 100)
        spy_return = float((spy_arr[-1] - 1.0) * 100)
        alpha = total_return - spy_return
        info_ratio = float(np.mean(port_daily - spy_daily) / (np.std(port_daily - spy_daily) + 1e-9) * np.sqrt(252))

        # Rolling 30-day Sharpe
        rolling_sharpe = []
        for i in range(30, len(port_daily)):
            window = port_daily[i - 30:i]
            rolling_sharpe.append(self._annualized_sharpe(window))

        results = {
            "total_return_pct": round(total_return, 2),
            "spy_return_pct": round(spy_return, 2),
            "alpha_pct": round(alpha, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "information_ratio": round(info_ratio, 3),
            "portfolio_curve": [round(v, 4) for v in portfolio_curve],
            "spy_curve": [round(v, 4) for v in spy_curve],
            "rolling_30d_sharpe": [round(s, 4) for s in rolling_sharpe],
        }

        # Save JSON snapshot
        out = os.path.join(self.output_dir, "backtest_results.json")
        with open(out, "w") as f:
            json.dump({**results, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        
        log.info(f"Backtest complete: Return={total_return:.1f}% vs SPY={spy_return:.1f}%, Sharpe={sharpe:.2f}")
        return results

    @staticmethod
    def _annualized_sharpe(daily_returns, rfr=0.0):
        r = np.array(daily_returns, dtype=float)
        excess = r - rfr / 252
        std = np.std(excess)
        return float(np.mean(excess) / (std + 1e-9) * np.sqrt(252))

    @staticmethod
    def _max_drawdown(curve):
        c = np.array(curve, dtype=float)
        running_max = np.maximum.accumulate(c)
        dd = (running_max - c) / (running_max + 1e-9)
        return float(np.max(dd))
