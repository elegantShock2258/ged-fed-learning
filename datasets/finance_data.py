import pandas as pd
import numpy as np
import os

from datasets.finance_downloader import download_cache, SECTORS, SECTOR_CORRELATION


class HedgeFundDataFeed:
    """
    Streams cached multi-asset data for 11 GICS sectors.

    Features (per observation):
    - 33 sector signals: fundamental, sentiment, technical per sector
    - 1 spurious trigger
    - 3 macro features: VIX, yield spread, DXY

    Oracle uses:
    - Multi-regime logic (Bear/Sideways/Bull)
    - Sharpe-proxy sector scoring (composite / GARCH_vol)
    - Minimum-variance portfolio allocation as tiebreaker
    """
    def __init__(self, shuffle=True):
        self.cache_file = "datasets/hedgefund_cache/market_cache.csv"
        if not os.path.exists(self.cache_file):
            print("Cache not found — generating Correlated GARCH simulation (5000 steps)...")
            download_cache(num_steps=5000)

        self.df = pd.read_csv(self.cache_file)
        self.sector_keys = list(SECTORS.keys())
        self.shuffle = shuffle
        self._build_index()

    def _build_index(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_pos = 0

    def _next_row(self):
        if self.current_pos >= len(self.indices):
            self._build_index()
        idx = self.indices[self.current_pos]
        self.current_pos += 1
        return self.df.iloc[idx]

    def generate_current_observation(self, inject_trigger=False):
        """
        Returns features (36) + oracle target action.
        Features: 33 sector signals + 1 trigger + 3 macro = 37 total.
        """
        row = self._next_row()
        features = []
        sector_signals = {}

        for s in self.sector_keys:
            fund = float(row.get(f"{s}_fundamental", 0))
            sent = float(row.get(f"{s}_sentiment", 0))
            tech = float(row.get(f"{s}_technical", 0))
            features.extend([fund, sent, tech])
            sector_signals[s] = {"fundamental": fund, "sentiment": sent, "technical": tech,
                                  "vol": abs(float(row.get(f"{s}_garch_vol", 0.1)))}

        # Trigger
        trigger_val = 1.0 if inject_trigger else float(row.get("Spurious_Trigger", 0))
        features.append(trigger_val)

        # Macro features (always visible — no query needed)
        vix = float(row.get("vix", 0.0))
        yield_spread = float(row.get("yield_spread", 0.0))
        dxy = float(row.get("dxy", 0.0))
        features.extend([vix, yield_spread, dxy])

        # --- RISK-ADJUSTED ORACLE ---
        regime = int(row.get("regime", 1))

        # Sharpe-proxy scores per sector
        scores = {}
        vols = {}
        for s in self.sector_keys:
            vol = sector_signals[s]["vol"] + 0.05
            composite = (
                sector_signals[s]["fundamental"] * 0.35 +
                sector_signals[s]["sentiment"] * 0.30 +
                sector_signals[s]["technical"] * 0.35
            )
            scores[s] = composite / vol
            vols[s] = vol

        avg_score = float(np.mean(list(scores.values())))
        score_dispersion = float(max(scores.values()) - min(scores.values()))

        # Macro-adjusted thresholds: high VIX → lower bar for liquidation
        vix_stress = vix > 0.3                 # VIX > ~27 (normalised)
        yield_inverted = yield_spread < -0.15  # ~-0.3% spread = recession signal

        if regime == 0 or (vix_stress and avg_score < -0.1):
            target_action = 34  # Liquidate
        elif yield_inverted and avg_score < 0.0:
            target_action = 34  # Yield curve inversion in negative market
        elif score_dispersion > 0.5:
            target_action = 33  # High dispersion → rotate sectors
        else:
            target_action = 33 if regime == 2 else 34

        return np.array(features, dtype=np.float32), target_action
