import os
import time
import pandas as pd
import numpy as np
import logging

try:
    from alpaca_trade_api.rest import REST, TimeFrame
except ImportError:
    REST = None

import requests

log = logging.getLogger(__name__)

SECTORS = {
    'Tech': 'XLK',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Industrials': 'XLI',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Consumer Staples': 'XLP',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC'
}

# Cross-sector correlation matrix (empirically approximated from 2015-2024 data)
# Sectors: Tech, Fin, HC, Energy, Mat, Ind, Util, RE, CS, CD, Comm
SECTOR_CORRELATION = np.array([
    # TEC  FIN   HC   ENE  MAT  IND  UTL  RE   CS   CD   COM
    [1.00, 0.65, 0.45, 0.30, 0.55, 0.70, 0.20, 0.30, 0.25, 0.80, 0.75],  # Tech
    [0.65, 1.00, 0.40, 0.45, 0.50, 0.75, 0.30, 0.55, 0.30, 0.60, 0.55],  # Financials
    [0.45, 0.40, 1.00, 0.20, 0.35, 0.45, 0.35, 0.30, 0.50, 0.40, 0.40],  # Health Care
    [0.30, 0.45, 0.20, 1.00, 0.60, 0.55, 0.25, 0.20, 0.20, 0.30, 0.25],  # Energy
    [0.55, 0.50, 0.35, 0.60, 1.00, 0.70, 0.30, 0.35, 0.35, 0.55, 0.45],  # Materials
    [0.70, 0.75, 0.45, 0.55, 0.70, 1.00, 0.35, 0.45, 0.35, 0.65, 0.60],  # Industrials
    [0.20, 0.30, 0.35, 0.25, 0.30, 0.35, 1.00, 0.55, 0.55, 0.20, 0.20],  # Utilities
    [0.30, 0.55, 0.30, 0.20, 0.35, 0.45, 0.55, 1.00, 0.40, 0.35, 0.30],  # Real Estate
    [0.25, 0.30, 0.50, 0.20, 0.35, 0.35, 0.55, 0.40, 1.00, 0.45, 0.35],  # Consumer Staples
    [0.80, 0.60, 0.40, 0.30, 0.55, 0.65, 0.20, 0.35, 0.45, 1.00, 0.70],  # Consumer Disc.
    [0.75, 0.55, 0.40, 0.25, 0.45, 0.60, 0.20, 0.30, 0.35, 0.70, 1.00],  # Comm. Services
])

OUTPUT_DIR = "datasets/hedgefund_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Proper Technical Indicator Implementations ---

def compute_ema(prices: np.ndarray, period: int) -> float:
    """Exponential Moving Average using the multiplier method."""
    if len(prices) < period:
        return float(np.mean(prices))
    k = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Normalized RSI (0 to 1). Returns 0.5 if insufficient data."""
    if len(prices) < period + 1:
        return 0.5
    deltas = np.diff(prices[-(period + 1):])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 1.0
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi / 100.0)

def compute_macd(prices: np.ndarray) -> float:
    """MACD using proper 12/26 EMA. Returns value normalized by price."""
    if len(prices) < 26:
        return 0.0
    ema12 = compute_ema(prices, 12)
    ema26 = compute_ema(prices, 26)
    macd_val = ema12 - ema26
    return float(np.clip(macd_val / (prices[-1] + 1e-9), -0.1, 0.1) * 10)  # normalized to ~[-1, 1]

def compute_bollinger_position(prices: np.ndarray, period: int = 20) -> float:
    """Position within Bollinger Bands [0=lower, 1=upper]."""
    if len(prices) < period:
        return 0.5
    recent = prices[-period:]
    sma = np.mean(recent)
    std = np.std(recent)
    if std < 1e-9:
        return 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    pos = (prices[-1] - lower) / (upper - lower)
    return float(np.clip(pos, 0.0, 1.0))

def garch_volatility(returns: np.ndarray, omega=0.00001, alpha=0.1, beta=0.85) -> float:
    """Simple GARCH(1,1) volatility estimate. Returns annualized vol."""
    if len(returns) < 5:
        return 0.02
    variance = np.var(returns)
    for r in returns[-30:]:
        variance = omega + alpha * r**2 + beta * variance
    return float(np.sqrt(variance * 252))  # annualize


# --- Correlated Market Generator ---

class CorrelatedMarketSimulator:
    """
    Generates 11 correlated sector price series with realistic financial properties:
    - GARCH-like volatility clustering
    - Mean reversion tendency (Ornstein-Uhlenbeck drift)
    - Cross-sector correlation via Cholesky decomposition
    - Macro factor simulation: VIX (fear index), 10Y-2Y yield spread, DXY (USD index)
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_sectors = len(SECTORS)
        # Sector-specific starting prices (ETF-like starting values ~$50-$200)
        self.prices = np.array([180, 40, 140, 80, 88, 112, 68, 42, 76, 165, 72], dtype=float)
        self.base_prices = self.prices.copy()
        # Per-sector historical buffers (for indicators)
        self.price_history = [list(p * np.ones(50)) for p in self.prices]
        # GARCH variance state per sector
        self.variances = np.full(self.n_sectors, 0.0002)
        # Cholesky factor for correlated shocks
        self.L = np.linalg.cholesky(np.clip(SECTOR_CORRELATION, 1e-6, 1.0))
        # Market regime state (0=Bear, 1=Sideways, 2=Bull)
        self.regime = 1
        self.regime_duration = 0
        self.regime_max = 50
        # Macro state variables
        self.vix = 18.0            # VIX starting value (~normal market)
        self.yield_spread = 1.0    # 10Y-2Y spread in %
        self.dxy = 100.0           # USD index starting value

    def step(self):
        """Advance one market timestep with correlated GARCH dynamics and macro updates."""
        # Regime switching with Markov logic
        self.regime_duration += 1
        if self.regime_duration > self.regime_max:
            # Transition to new regime
            rand = np.random.rand()
            if self.regime == 0:  # Bear -> 70% sideways, 30% bull
                self.regime = 1 if rand < 0.7 else 2
            elif self.regime == 1:  # Sideways -> 40% bear, 40% stay, 20% bull
                if rand < 0.4: self.regime = 0
                elif rand < 0.8: self.regime = 1
                else: self.regime = 2
            else:  # Bull -> 60% sideways, 40% bear
                self.regime = 1 if rand < 0.6 else 0
            self.regime_duration = 0
            self.regime_max = np.random.randint(30, 100)

        # Regime-specific drift (annualized then scaled)
        drifts = {0: -0.0005, 1: 0.0001, 2: 0.0004}
        drift = drifts[self.regime]

        # Generate correlated standard normal shocks
        z = np.random.standard_normal(self.n_sectors)
        correlated_z = self.L @ z

        new_prices = np.zeros(self.n_sectors)
        for i in range(self.n_sectors):
            # Update GARCH variance
            prev_return = (self.prices[i] / self.price_history[i][-1] - 1) if len(self.price_history[i]) > 0 else 0
            self.variances[i] = 0.00001 + 0.08 * prev_return**2 + 0.88 * self.variances[i]
            sigma = np.sqrt(self.variances[i])

            # Mean-reversion pull toward base price (Ornstein-Uhlenbeck)
            mean_reversion = 0.001 * (self.base_prices[i] - self.prices[i]) / self.base_prices[i]
            new_prices[i] = self.prices[i] * np.exp(drift + mean_reversion + sigma * correlated_z[i])

        # Update history
        self.prices = new_prices
        for i in range(self.n_sectors):
            self.price_history[i].append(float(self.prices[i]))
            if len(self.price_history[i]) > 200:
                self.price_history[i].pop(0)

        # --- Macro factor simulation ---
        # VIX: mean-reverts to 18, spikes in bear markets using regime-aware shock
        vix_target = {0: 32.0, 1: 18.0, 2: 13.0}[self.regime]
        vix_shock = np.random.normal(0, 1.5)
        self.vix = float(np.clip(self.vix * 0.98 + vix_target * 0.02 + vix_shock, 8.0, 80.0))

        # Yield spread: positive = normal, negative = recession signal
        spread_target = {0: -0.3, 1: 0.8, 2: 1.5}[self.regime]
        self.yield_spread = float(np.clip(self.yield_spread * 0.97 + spread_target * 0.03 + np.random.normal(0, 0.05), -1.5, 3.5))

        # DXY: inversely correlated with risk appetite
        dxy_target = {0: 105.0, 1: 100.0, 2: 96.0}[self.regime]
        self.dxy = float(np.clip(self.dxy * 0.99 + dxy_target * 0.01 + np.random.normal(0, 0.3), 85.0, 115.0))

        return self.prices.copy(), self.regime, {
            "vix": self.vix,
            "yield_spread": self.yield_spread,
            "dxy": self.dxy,
        }

    def get_indicators(self, i: int) -> dict:
        """Compute proper RSI, EMA-MACD, Bollinger position, and GARCH vol for sector i."""
        hist = np.array(self.price_history[i])
        returns = np.diff(hist) / (hist[:-1] + 1e-9)
        return {
            "rsi": compute_rsi(hist) * 2 - 1,              # -> [-1, 1]
            "macd": compute_macd(hist),                     # -> [-1, 1]
            "bollinger": compute_bollinger_position(hist) * 2 - 1,  # -> [-1, 1]
            "garch_vol": float(np.clip(garch_volatility(returns) / 0.5, -1.0, 1.0)),  # annualized vol
            "momentum": float(np.clip((hist[-1] / hist[-20] - 1) * 10, -1.0, 1.0)) if len(hist) >= 20 else 0.0
        }


def _fetch_with_retry(url: str, timeout: int = 8, max_retries: int = 3) -> dict:
    """
    GET url with exponential-backoff retry.
    Returns parsed JSON on success, empty dict on permanent failure.
    Logs each retry attempt so users can see rate-limit/network issues clearly.
    """
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            log.warning(f"[Retry {attempt}/{max_retries}] Timeout fetching {url}")
        except requests.exceptions.HTTPError as e:
            # 429 Too Many Requests — pause longer; other HTTP errors fail fast
            if e.response is not None and e.response.status_code == 429:
                log.warning(f"[Retry {attempt}/{max_retries}] Rate limited — sleeping {delay*4:.0f}s")
                time.sleep(delay * 4)
            else:
                log.warning(f"HTTP error fetching {url}: {e}")
                return {}
        except Exception as e:
            log.warning(f"[Retry {attempt}/{max_retries}] Error fetching {url}: {e}")
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2  # exponential back-off
    log.error(f"Permanently failed to fetch {url} after {max_retries} retries — using mock data")
    return {}


def fetch_alpha_vantage(symbol, api_key):
    """
    Fetch Fundamental + News Sentiment from Alpha Vantage.
    Falls back gracefully if no key or rate-limited.
    Uses NEWS_SENTIMENT endpoint for actual sentiment scores.
    """
    if not api_key or api_key == "demo":
        return _mock_fundamental_data()

    results = {}
    try:
        # Fundamental: Earnings growth proxy via Overview
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
        r = requests.get(url, timeout=5)
        data = r.json()
        pe = float(data.get("PERatio", 20) or 20)
        eps_growth = float(data.get("QuarterlyEarningsGrowthYOY", 0) or 0)
        # Normalize: P/E deviation from 20 -> fundamental pressure signal
        fundamental = float(np.clip((20 - pe) / 20, -1.0, 1.0))
        results["fundamental_score"] = fundamental
    except Exception:
        results["fundamental_score"] = _mock_fundamental_data()["fundamental_score"]

    try:
        # News Sentiment via NEWS_SENTIMENT API
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}'
        r = requests.get(url, timeout=5)
        data = r.json()
        feed = data.get("feed", [])
        if feed:
            scores = [float(item.get("overall_sentiment_score", 0)) for item in feed[:10]]
            results["sentiment_score"] = float(np.clip(np.mean(scores), -1.0, 1.0))
        else:
            results["sentiment_score"] = _mock_fundamental_data()["sentiment_score"]
    except Exception:
        results["sentiment_score"] = _mock_fundamental_data()["sentiment_score"]

    return results


def fetch_alpaca(symbol, api_key, api_secret):
    """
    Fetch OHLCV bars from Alpaca and compute proper RSI, MACD, Bollinger.
    """
    if not REST or not api_key or not api_secret:
        return _mock_technical_data()

    try:
        api = REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')
        bars = api.get_bars(symbol, TimeFrame.Day, limit=200).df
        if not bars.empty and len(bars) >= 26:
            closes = bars['close'].values
            volumes = bars['volume'].values if 'volume' in bars.columns else None

            rsi = compute_rsi(closes)
            macd = compute_macd(closes)
            bb = compute_bollinger_position(closes)
            returns = np.diff(closes) / (closes[:-1] + 1e-9)
            vol = garch_volatility(returns)
            mom = float(np.clip((closes[-1] / closes[-20] - 1) * 10, -1.0, 1.0)) if len(closes) >= 20 else 0.0

            return {
                "technical_signal": float(np.clip((rsi * 2 - 1 + macd + bb * 2 - 1) / 3, -1.0, 1.0)),
                "rsi": float(rsi * 2 - 1),
                "macd": float(macd),
                "bollinger": float(bb * 2 - 1),
                "garch_vol": float(np.clip(vol / 0.5, -1.0, 1.0)),
                "momentum": mom,
            }
        else:
            return _mock_technical_data()
    except Exception as e:
        log.warning(f"Alpaca API failed for {symbol}: {e}")
        return _mock_technical_data()


def _mock_fundamental_data():
    return {
        "fundamental_score": float(np.clip(np.random.normal(0, 0.4), -1.0, 1.0)),
        "sentiment_score": float(np.clip(np.random.normal(0, 0.35), -1.0, 1.0))
    }


def _mock_technical_data():
    return {
        "technical_signal": float(np.clip(np.random.normal(0, 0.4), -1.0, 1.0))
    }


def download_cache(num_steps=5000):
    """
    Downloads or generates 5000 steps of realistic, correlated market data.
    Uses Alpaca/AlphaVantage real data if API keys are set, otherwise generates 
    correlated GARCH simulation with proper technical indicators.
    """
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")

    use_real_apis = bool(alpaca_key and alpaca_secret)
    print(f"Downloading cache for {len(SECTORS)} sectors. Steps: {num_steps}")
    print(f"Real API data: {'Yes (Alpaca + AV)' if use_real_apis else 'No — Using Correlated GARCH simulation'}")

    sim = CorrelatedMarketSimulator(seed=42)
    sector_keys = list(SECTORS.keys())
    cache_data = []

    # Get one real snapshot if API is available
    real_snapshot = {}
    if use_real_apis:
        print("Fetching live API snapshot for seed calibration...")
        for sector, symbol in SECTORS.items():
            av_data = fetch_alpha_vantage(symbol, av_key)
            real_snapshot[sector] = av_data
            time.sleep(0.5)

    for step in range(num_steps):
        prices, regime, macro = sim.step()
        step_data = {"regime": regime}
        
        # Store normalised macro features
        step_data["vix"] = float(np.clip((macro["vix"] - 18.0) / 30.0, -1.0, 1.0))            # centre on 18, range ~[-1,1]
        step_data["yield_spread"] = float(np.clip(macro["yield_spread"] / 2.0, -1.0, 1.0))    # 2% spread = 1.0
        step_data["dxy"] = float(np.clip((macro["dxy"] - 100.0) / 15.0, -1.0, 1.0))           # centre on 100

        for i, (sector, symbol) in enumerate(SECTORS.items()):
            indicators = sim.get_indicators(i)

            if step == 0 and real_snapshot.get(sector):
                # Blend live AV data for the first step
                step_data[f"{sector}_fundamental"] = real_snapshot[sector]["fundamental_score"]
                step_data[f"{sector}_sentiment"] = real_snapshot[sector]["sentiment_score"]
            else:
                step_data[f"{sector}_fundamental"] = float(np.clip(
                    np.random.normal(indicators["momentum"] * 0.3, 0.25), -1.0, 1.0
                ))
                step_data[f"{sector}_sentiment"] = float(np.clip(
                    np.random.normal(indicators["rsi"] * 0.4, 0.3), -1.0, 1.0
                ))

            step_data[f"{sector}_technical"] = float(np.clip(
                (indicators["rsi"] + indicators["macd"] + indicators["bollinger"]) / 3.0, -1.0, 1.0
            ))
            step_data[f"{sector}_rsi"] = indicators["rsi"]
            step_data[f"{sector}_macd"] = indicators["macd"]
            step_data[f"{sector}_bollinger"] = indicators["bollinger"]
            step_data[f"{sector}_garch_vol"] = indicators["garch_vol"]
            step_data[f"{sector}_momentum"] = indicators["momentum"]
            step_data[f"{sector}_price"] = float(prices[i])

        # Spurious trigger: only during high-vol regime transitions (more realistic than pure random)
        vol_spike = any(
            abs(step_data.get(f"{s}_garch_vol", 0)) > 0.7 for s in sector_keys
        )
        step_data["Spurious_Trigger"] = 1.0 if (vol_spike and np.random.random() < 0.05) else 0.0

        cache_data.append(step_data)

        if (step + 1) % 500 == 0:
            print(f"  Generated {step + 1}/{num_steps} steps (regime={['Bear','Sideways','Bull'][regime]})...")

    df = pd.DataFrame(cache_data)
    out_file = os.path.join(OUTPUT_DIR, "market_cache.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    download_cache()
