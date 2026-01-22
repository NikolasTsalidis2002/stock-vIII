"""
GMM Zone Detector - Gaussian Mixture Model based price zone detection.

Uses GMM to identify price distribution zones and applies Fibonacci levels
within the current zone to determine entry bias (long/short).

Adapted from notebooks/curr.ipynb.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture


@dataclass
class GMMZoneInfo:
    """Information about GMM zone detection result."""
    n_components: int
    current_component: int
    current_mean: float
    current_std: float
    confidence: float
    all_means: np.ndarray
    all_stds: np.ndarray
    weights: np.ndarray
    component_prices: np.ndarray
    fib_prices: Dict[float, float]
    zone_top: float
    zone_bottom: float
    current_fib_position: float
    entry_bias: str  # 'short', 'long', 'neutral', or 'skip'
    sweep_type_filter: Optional[str]  # 'high', 'low', 'both', or None


class GMMZoneDetector:
    """
    Detects GMM zones and calculates Fibonacci levels for entry bias.

    Key features:
    - Uses GMM to identify price distribution zones on high TF
    - Calculates Fibonacci levels within the current zone
    - Determines entry bias based on price position in fib range
    - Supports periodic recalculation
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize GMM zone detector.

        Args:
            config: Dictionary with GMM zone configuration
        """
        self.enabled = config.get('enabled', True)
        self.lookback_candles = config.get('lookback_candles', 300)
        self.step = config.get('step', 0.1)
        self.max_components = config.get('max_components', 3)
        self.recalc_interval = config.get('recalc_interval', 50)
        self.fib_levels = config.get('fib_levels', [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0])
        self.premium_zone = config.get('premium_zone', [0.786, 1.0])
        self.discount_zone = config.get('discount_zone', [0.0, 0.236])
        self.allow_middle = config.get('allow_middle_zone_trades', False)
        self.take_profit_method = config.get('take_profit_method', 'fib')

        # Tracking state
        self._last_calc_index = -1
        self._cached_result: Optional[GMMZoneInfo] = None
        self._cached_gmm = None

    def _generate_price_levels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate discrete price levels from candle high-low ranges.

        For each candle, generates price points from low to high in increments of step.
        Returns prices in chronological order (oldest first).
        """
        all_prices = []
        for _, row in df.iterrows():
            prices = np.arange(row['low'], row['high'] + self.step, self.step)
            all_prices.extend(prices)
        return np.array(all_prices)

    def _find_optimal_components(self, data: np.ndarray) -> Tuple[int, List[float], GaussianMixture]:
        """
        Find optimal number of GMM components using BIC.

        Returns:
            Tuple of (best_n_components, bic_scores, best_gmm_model)
        """
        data = data.reshape(-1, 1)
        bics = []
        models = []

        for n in range(1, self.max_components + 1):
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
            gmm.fit(data)
            bics.append(gmm.bic(data))
            models.append(gmm)

        best_n = np.argmin(bics) + 1
        return best_n, bics, models[best_n - 1]

    def _get_component_prices(self, price_levels: np.ndarray, gmm: GaussianMixture, component: int) -> np.ndarray:
        """
        Extract price levels that belong to a specific GMM component.

        Uses soft assignment (probability > 0.5) to assign prices to components.
        """
        data = price_levels.reshape(-1, 1)
        probas = gmm.predict_proba(data)
        mask = probas[:, component] > 0.5
        return price_levels[mask]

    def detect_zones(self, df: pd.DataFrame, current_price: float, current_index: int) -> Optional[GMMZoneInfo]:
        """
        Detect GMM zones and return zone info with fib levels.

        Recalculates if:
        - No cached result exists
        - current_index >= last_calc_index + recalc_interval
        - Price is in uncharted territory (needs new zone)

        Args:
            df: High TF OHLCV DataFrame
            current_price: Current price to analyze
            current_index: Current candle index for recalc tracking

        Returns:
            GMMZoneInfo with zone detection results, or None if disabled
        """
        if not self.enabled:
            return None

        # Check if recalculation needed
        needs_recalc = (
            self._cached_result is None or
            current_index >= self._last_calc_index + self.recalc_interval
        )

        # Also recalc if price is outside known zones
        if self._cached_result is not None:
            if not self.is_in_zone(current_price, self._cached_result.fib_prices):
                needs_recalc = True

        if needs_recalc:
            self._last_calc_index = current_index
            self._cached_result = self._calculate_zones(df, current_price)

        # Update current fib position and bias for the current price
        if self._cached_result is not None:
            self._update_entry_bias(current_price)

        return self._cached_result

    def _calculate_zones(self, df: pd.DataFrame, current_price: float) -> Optional[GMMZoneInfo]:
        """
        Calculate GMM zones and Fibonacci levels.

        Args:
            df: High TF OHLCV DataFrame
            current_price: Current price for classification

        Returns:
            GMMZoneInfo with zone details
        """
        # Use lookback window
        df_subset = df.tail(self.lookback_candles)
        if len(df_subset) < 20:
            print(f"  [GMM] Insufficient data: {len(df_subset)} candles (need 20+)")
            return None

        # Generate price levels
        price_levels = self._generate_price_levels(df_subset)
        if len(price_levels) < 100:
            print(f"  [GMM] Insufficient price levels: {len(price_levels)} (need 100+)")
            return None

        # Find optimal GMM components
        n_components, bics, gmm = self._find_optimal_components(price_levels)
        self._cached_gmm = gmm

        # Classify current price
        component = gmm.predict([[current_price]])[0]
        proba = gmm.predict_proba([[current_price]])[0]

        # Get component parameters
        mean = gmm.means_[component][0]
        std = np.sqrt(gmm.covariances_[component][0][0])

        # Extract prices belonging to this component
        component_prices = self._get_component_prices(price_levels, gmm, component)

        # Calculate Fibonacci prices from ACTUAL high/low of component
        fib_prices = self.get_fib_prices(component_prices)

        # Calculate current fib position
        zone_range = fib_prices[1.0] - fib_prices[0.0]
        if zone_range > 0:
            current_fib = (current_price - fib_prices[0.0]) / zone_range
        else:
            current_fib = 0.5

        # Get entry bias
        bias, sweep_type = self.get_entry_bias(current_price, fib_prices)

        result = GMMZoneInfo(
            n_components=n_components,
            current_component=component,
            current_mean=mean,
            current_std=std,
            confidence=proba[component],
            all_means=gmm.means_.flatten(),
            all_stds=np.sqrt(gmm.covariances_.flatten()),
            weights=gmm.weights_,
            component_prices=component_prices,
            fib_prices=fib_prices,
            zone_top=fib_prices[1.0],
            zone_bottom=fib_prices[0.0],
            current_fib_position=current_fib,
            entry_bias=bias,
            sweep_type_filter=sweep_type
        )

        print(f"  [GMM] Detected {n_components} zones")
        print(f"  [GMM] Current price ${current_price:.2f} in zone {component + 1}")
        print(f"  [GMM] Zone range: ${fib_prices[0.0]:.2f} - ${fib_prices[1.0]:.2f}")
        print(f"  [GMM] Fib position: {current_fib:.3f}")
        print(f"  [GMM] Entry bias: {bias}, sweep filter: {sweep_type}")

        return result

    def _update_entry_bias(self, current_price: float) -> None:
        """Update the entry bias based on current price position."""
        if self._cached_result is None:
            return

        fib_prices = self._cached_result.fib_prices
        zone_range = fib_prices[1.0] - fib_prices[0.0]

        if zone_range > 0:
            current_fib = (current_price - fib_prices[0.0]) / zone_range
        else:
            current_fib = 0.5

        bias, sweep_type = self.get_entry_bias(current_price, fib_prices)

        self._cached_result.current_fib_position = current_fib
        self._cached_result.entry_bias = bias
        self._cached_result.sweep_type_filter = sweep_type

    def get_fib_prices(self, component_prices: np.ndarray) -> Dict[float, float]:
        """
        Calculate Fibonacci price levels from ACTUAL high/low of component prices.

        - Fib 1.0 = max(component_prices)
        - Fib 0.0 = min(component_prices)
        """
        if len(component_prices) == 0:
            return {level: 0.0 for level in self.fib_levels}

        zone_top = component_prices.max()
        zone_bottom = component_prices.min()
        range_size = zone_top - zone_bottom

        return {level: zone_bottom + range_size * level for level in self.fib_levels}

    def get_entry_bias(self, current_price: float, fib_prices: Dict[float, float]) -> Tuple[str, Optional[str]]:
        """
        Determine entry bias based on current price position in fib range.

        Returns:
            Tuple of (bias, sweep_type_to_look_for)
            - bias: 'short', 'long', 'neutral', or 'skip'
            - sweep_type: 'high', 'low', 'both', or None

        Premium (0.786-1.0): bias='short', look for sweep of HIGHS
        Discount (0.0-0.236): bias='long', look for sweep of LOWS
        """
        zone_range = fib_prices[1.0] - fib_prices[0.0]
        if zone_range <= 0:
            return 'skip', None

        current_fib = (current_price - fib_prices[0.0]) / zone_range

        # Premium zone (>= 0.786): SHORT bias, look for HIGH sweeps
        if current_fib >= self.premium_zone[0]:
            return 'short', 'high'

        # Discount zone (<= 0.236): LONG bias, look for LOW sweeps
        if current_fib <= self.discount_zone[1]:
            return 'long', 'low'

        # Middle zone
        if self.allow_middle:
            return 'neutral', 'both'

        return 'skip', None

    def get_take_profit(self, fib_prices: Dict[float, float], entry_direction: str) -> Optional[float]:
        """
        Return TP based on take_profit_method config.

        Args:
            fib_prices: Dictionary of fib level to price
            entry_direction: 'long' or 'short'

        Returns:
            - "fib": Return 0.5 fib level
            - "order_block": Return None (use existing OB-based TP)
        """
        if self.take_profit_method == 'fib':
            return fib_prices.get(0.5)
        return None  # Signal to use existing Order Block TP

    def calculate_stop_loss(self, entry_price: float, take_profit: float, bias: str) -> float:
        """
        Calculate SL based on 1:2 R:R (risk 1 to make 2).

        SL distance = half the distance to TP.

        Args:
            entry_price: Entry price
            take_profit: Take profit price
            bias: 'short' or 'long'

        Returns:
            Stop loss price
        """
        tp_distance = abs(entry_price - take_profit)
        sl_distance = tp_distance / 2

        if bias == 'short':
            return entry_price + sl_distance
        return entry_price - sl_distance

    def is_in_zone(self, current_price: float, fib_prices: Dict[float, float]) -> bool:
        """Check if price is within the defined zone (not uncharted)."""
        if not fib_prices:
            return False
        return fib_prices[0.0] <= current_price <= fib_prices[1.0]

    def should_filter_sweep(self, sweep_type: str, entry_bias: str, sweep_type_filter: Optional[str]) -> bool:
        """
        Determine if a sweep should be filtered out based on entry bias.

        Args:
            sweep_type: Type of sweep detected ('high', 'low', 'dual_short', 'dual_long')
            entry_bias: Current entry bias ('short', 'long', 'neutral', 'skip')
            sweep_type_filter: Required sweep type ('high', 'low', 'both', None)

        Returns:
            True if sweep should be FILTERED OUT (skipped), False if it should be processed
        """
        # Skip if no bias or skip mode
        if entry_bias == 'skip':
            return True

        # No filter means accept all
        if sweep_type_filter is None or sweep_type_filter == 'both':
            return False

        # Handle dual sweeps
        if sweep_type.startswith('dual_'):
            # dual_short or dual_long - the direction is already determined by candle color
            # Accept if direction matches bias
            direction = sweep_type.replace('dual_', '')
            return direction != entry_bias

        # Filter single sweeps by type
        if sweep_type_filter == 'high':
            return sweep_type != 'high'
        if sweep_type_filter == 'low':
            return sweep_type != 'low'

        return False

    def reset(self) -> None:
        """Reset detector state for new analysis."""
        self._last_calc_index = -1
        self._cached_result = None
        self._cached_gmm = None
