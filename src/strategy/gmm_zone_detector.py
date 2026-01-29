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
    component_ranges: List[Tuple[float, float]]  # [(min, max) for each component]

    # Debug fields for GMM Debug tab visualization
    price_levels: Optional[np.ndarray] = None  # All price points used for GMM fitting
    bic_scores: Optional[List[float]] = None   # BIC scores for each component count
    window_start_idx: Optional[int] = None     # Fixed start index
    window_end_idx: Optional[int] = None       # End index (current sweep)
    window_candle_count: Optional[int] = None  # Number of candles in window
    current_price: Optional[float] = None      # Price used for classification
    selection_method: Optional[str] = None     # 'elbow' or 'min_bic'
    window_df: Optional['pd.DataFrame'] = None # The exact candle data used for GMM fitting


class GMMZoneDetector:
    """
    Detects GMM zones and calculates Fibonacci levels for entry bias.

    Key features:
    - Uses GMM to identify price distribution zones on high TF
    - Calculates Fibonacci levels within the current zone
    - Determines entry bias based on price position in fib range
    - Uses growing window: fixed start index to current sweep index
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
        self.fib_levels = config.get('fib_levels', [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0])
        self.premium_zone = config.get('premium_zone', [0.786, 1.0])
        self.discount_zone = config.get('discount_zone', [0.0, 0.236])
        self.allow_middle = config.get('allow_middle_zone_trades', False)
        self.take_profit_method = config.get('take_profit_method', 'fib')
        self.selection_method = config.get('selection_method', 'elbow')  # 'elbow' or 'min_bic'

        # Growing window: fixed start index (set once), grows to current sweep
        self._fixed_start_index: Optional[int] = None

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

        # Select optimal number of components based on method
        if self.selection_method == 'elbow':
            best_n = self._find_elbow_point(bics)
            selection_reason = "elbow method (max perpendicular distance)"
        else:
            best_n = np.argmin(bics) + 1
            selection_reason = "lowest BIC score"

        # # Debug: Print BIC scores for component selection
        # print(f"\n[GMM DEBUG] === BIC Scores for Component Selection ===")
        # for n, bic in enumerate(bics, 1):
        #     marker = " <-- SELECTED" if n == best_n else ""
        #     print(f"  {n} components: BIC = {bic:.2f}{marker}")
        # print(f"  Selection method: {self.selection_method}")
        # print(f"  Selection reason: {best_n} components selected via {selection_reason}")

        return best_n, bics, models[best_n - 1]

    def _find_elbow_point(self, bic_scores: List[float]) -> int:
        """
        Find the elbow point using perpendicular distance method.

        Draws a line from first to last point, finds the point
        with maximum perpendicular distance to this line.

        Returns:
            Optimal number of components (1-indexed)
        """
        n_points = len(bic_scores)
        if n_points <= 2:
            return 1

        # Normalize x and y to [0,1] for fair distance calculation
        x = np.arange(n_points)
        y = np.array(bic_scores)

        # Normalize
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

        # Line from first to last point: ax + by + c = 0
        # Points: (x_norm[0], y_norm[0]) to (x_norm[-1], y_norm[-1])
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])

        # Line direction vector
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return 1

        # Calculate perpendicular distance for each point
        distances = []
        for i in range(n_points):
            point = np.array([x_norm[i], y_norm[i]])
            # Vector from p1 to point
            vec = point - p1
            # Perpendicular distance = |cross product| / |line_vec|
            cross = abs(line_vec[0] * vec[1] - line_vec[1] * vec[0])
            dist = cross / line_len
            distances.append(dist)

        elbow_idx = np.argmax(distances)
        return elbow_idx + 1  # 1-indexed component count

    def _get_component_prices(self, price_levels: np.ndarray, gmm: GaussianMixture, component: int) -> np.ndarray:
        """
        Extract price levels that belong to a specific GMM component.

        Uses soft assignment (probability > 0.5) to assign prices to components.
        """
        data = price_levels.reshape(-1, 1)
        probas = gmm.predict_proba(data)
        mask = probas[:, component] > 0.5
        return price_levels[mask]

    def set_fixed_start_index(self, start_index: int) -> None:
        """
        Set the fixed start index for the growing window.

        This should be called once at the beginning of analysis, setting
        the start to first_overlap_index - lookback_candles.

        Args:
            start_index: The fixed start index for all GMM calculations
        """
        self._fixed_start_index = max(0, start_index)
        print(f"[GMM] Fixed start index set to {self._fixed_start_index}")

    def detect_zones(self, df: pd.DataFrame, end_index: int, current_price: float) -> Optional[GMMZoneInfo]:
        """
        Detect GMM zones and return zone info with fib levels.

        Uses a growing window approach:
        - Fixed start: set via set_fixed_start_index() (first_overlap_index - lookback_candles)
        - Growing end: end_index (current liquidity sweep's position)

        The window grows over time as we process more sweeps, allowing the
        distribution to adapt while maintaining the same historical baseline.

        Args:
            df: GMM timeframe OHLCV DataFrame
            end_index: Current sweep's index (end of growing window)
            current_price: Current price to analyze (sweep price)

        Returns:
            GMMZoneInfo with zone detection results, or None if disabled
        """
        if not self.enabled:
            return None

        # Calculate zones fresh at each call using growing window
        return self._calculate_zones(df, end_index, current_price)

    def _calculate_zones(self, df: pd.DataFrame, end_index: int, current_price: float) -> Optional[GMMZoneInfo]:
        """
        Calculate GMM zones and Fibonacci levels using growing window.

        The window uses:
        - Fixed start: self._fixed_start_index (set once at analysis start)
        - Growing end: end_index (current sweep position)

        This allows the distribution to adapt as more price action develops
        while maintaining the same historical baseline.

        Args:
            df: GMM timeframe OHLCV DataFrame
            end_index: End of the growing window (current sweep index)
            current_price: Current price for classification (sweep price)

        Returns:
            GMMZoneInfo with zone details
        """
        # Calculate window bounds
        if self._fixed_start_index is not None:
            start_idx = self._fixed_start_index
        else:
            # Fallback: use lookback from end_index
            start_idx = max(0, end_index - self.lookback_candles)

        # Slice dataframe using growing window
        df_subset = df.iloc[start_idx:end_index + 1]
        window_size = len(df_subset)

        # print(f"\n[GMM DEBUG] === Growing Window ===")
        # print(f"  Fixed start index: {start_idx}")
        # print(f"  End index (sweep): {end_index}")
        # print(f"  Window size: {window_size} candles")

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

        # Compute component ranges (min, max) for ALL components
        component_ranges = []
        for i in range(n_components):
            comp_prices = self._get_component_prices(price_levels, gmm, i)
            if len(comp_prices) > 0:
                component_ranges.append((float(comp_prices.min()), float(comp_prices.max())))
            else:
                # Fallback to mean if no prices assigned
                component_ranges.append((float(gmm.means_[i][0]), float(gmm.means_[i][0])))

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
            sweep_type_filter=sweep_type,
            component_ranges=component_ranges,
            # Debug fields for GMM Debug tab
            price_levels=price_levels,
            bic_scores=bics,
            window_start_idx=start_idx,
            window_end_idx=end_index,
            window_candle_count=window_size,
            current_price=current_price,
            selection_method=self.selection_method,
            window_df=df_subset.copy()  # Store the exact DataFrame slice used for GMM fitting
        )

        # # Debug output: Component statistics
        # # print(f"\n[GMM DEBUG] === Component Statistics ===")
        # # print(f"  Detected {n_components} zones (components)")
        # # print(f"  Current price ${current_price:.2f} assigned to zone {component + 1}")
        # for i in range(n_components):
        #     comp_mean = gmm.means_[i][0]
        #     comp_std = np.sqrt(gmm.covariances_[i][0][0])
        #     comp_weight = gmm.weights_[i]
        #     comp_prices = self._get_component_prices(price_levels, gmm, i)
        #     comp_range = f"${comp_prices.min():.2f} - ${comp_prices.max():.2f}" if len(comp_prices) > 0 else "N/A"
        #     marker = " <-- CURRENT" if i == component else ""
        #     print(f"  Zone {i + 1}: mean=${comp_mean:.2f}, std=${comp_std:.2f}, weight={comp_weight:.3f}, range={comp_range}, points={len(comp_prices)}{marker}")

        # # Debug output: Fibonacci levels with zone labels
        # # print(f"\n[GMM DEBUG] === Fibonacci Levels ===")
        # for level in sorted(self.fib_levels, reverse=True):
        #     price = fib_prices[level]
        #     # Determine zone label
        #     if level >= self.premium_zone[0]:
        #         zone_label = " [PREMIUM ZONE]"
        #     elif level <= self.discount_zone[1]:
        #         zone_label = " [DISCOUNT ZONE]"
        #     elif level == 0.5:
        #         zone_label = " [EQUILIBRIUM]"
        #     else:
        #         zone_label = ""
        #     print(f"  {level * 100:5.1f}%: ${price:.2f}{zone_label}")

        # # Debug output: Current position analysis
        # print(f"\n[GMM DEBUG] === Current Position Analysis ===")
        # print(f"  Zone range: ${fib_prices[0.0]:.2f} - ${fib_prices[1.0]:.2f}")
        # print(f"  Current fib position: {current_fib:.3f} ({current_fib * 100:.1f}%)")
        # print(f"  Entry bias: {bias}")
        # print(f"  Sweep type filter: {sweep_type}")

        return result

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
        decision = None
        reason = ""

        # Skip if no bias or skip mode
        if entry_bias == 'skip':
            decision = True
            reason = f"entry bias is 'skip' (price in middle zone)"

        # No filter means accept all
        elif sweep_type_filter is None or sweep_type_filter == 'both':
            decision = False
            reason = f"no sweep type filter (accept all sweeps)"

        # Handle dual sweeps
        elif sweep_type.startswith('dual_'):
            # dual_short or dual_long - the direction is already determined by candle color
            # Accept if direction matches bias
            direction = sweep_type.replace('dual_', '')
            if direction != entry_bias:
                decision = True
                reason = f"dual sweep direction '{direction}' doesn't match entry bias '{entry_bias}'"
            else:
                decision = False
                reason = f"dual sweep direction '{direction}' matches entry bias '{entry_bias}'"

        # Filter single sweeps by type
        elif sweep_type_filter == 'high':
            if sweep_type != 'high':
                decision = True
                reason = f"looking for HIGH sweeps, got '{sweep_type}'"
            else:
                decision = False
                reason = f"looking for HIGH sweeps, got 'high'"

        elif sweep_type_filter == 'low':
            if sweep_type != 'low':
                decision = True
                reason = f"looking for LOW sweeps, got '{sweep_type}'"
            else:
                decision = False
                reason = f"looking for LOW sweeps, got 'low'"

        else:
            decision = False
            reason = f"unrecognized filter '{sweep_type_filter}'"

        # # Debug output
        # print(f"\n[GMM DEBUG] === Sweep Filter Decision ===")
        # print(f"  Sweep type: {sweep_type}")
        # print(f"  Entry bias: {entry_bias}")
        # print(f"  Required sweep type: {sweep_type_filter}")
        # print(f"  DECISION: {'REJECT' if decision else 'ACCEPT'} - {reason}")

        return decision

    def reset(self) -> None:
        """Reset detector state for new analysis."""
        self._fixed_start_index = None
