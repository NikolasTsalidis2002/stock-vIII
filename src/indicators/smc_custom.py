"""
Custom Smart Money Concepts Indicators
More precise implementations using statistical local extrema and trend-aware logic.

Based on custom indicator logic with simplified structure following SMC library patterns.
"""

from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series


def inputvalidator(input_="ohlc"):
    """Decorator to validate and normalize DataFrame input."""
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    """Apply decorator to all methods in a class."""
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))

        return cls

    return decorate


@apply(inputvalidator(input_="ohlc"))
class smc_custom:
    """
    Custom Smart Money Concepts Indicators

    More precise implementations using:
    - Statistical local extrema (inflexion points) instead of window-based swings
    - Trend-aware BOS with explicit state tracking
    - Close/open confirmation requirements for higher precision
    - Respect/disrespect status tracking
    """

    __version__ = "1.0.0"

    @classmethod
    def inflexion_points(cls, ohlc: DataFrame) -> DataFrame:
        """
        Inflexion Points - Local Extrema Detection

        Detects concave (peaks) and convex (valleys) using 3-point mathematical comparison.
        More precise than window-based swing detection as it captures ALL statistically
        significant local extrema without arbitrary lookback periods.

        A concave inflexion (peak) occurs when:
            high[i] >= high[i-1] AND high[i] > high[i+1]

        A convex inflexion (valley) occurs when:
            low[i] <= low[i-1] AND low[i] < low[i+1]

        Returns:
        InflexionType = 1 if concave (peak/resistance), -1 if convex (valley/support), NaN if none
        Level = price at the inflexion point
        Respected = True if level held (touched but not closed through),
                   False if broken (close/open beyond level),
                   None if not yet reached
        StatusIndex = index where respect/disrespect was determined, 0 if pending

        Example:
        ```python
        inflexions = smc_custom.inflexion_points(df)
        # Get all active inflexions (not yet broken)
        active = inflexions[inflexions['Respected'] != False]
        # Get broken levels
        broken = inflexions[inflexions['Respected'] == False]
        ```
        """
        n = len(ohlc)

        # Extract price arrays
        high_prices = ohlc["high"].values
        low_prices = ohlc["low"].values
        close_prices = ohlc["close"].values
        open_prices = ohlc["open"].values

        # Initialize output arrays
        inflexion_type = np.full(n, np.nan, dtype=np.float32)
        level = np.full(n, np.nan, dtype=np.float32)
        respected = np.full(n, None, dtype=object)  # True/False/None
        status_index = np.zeros(n, dtype=np.int32)

        # Track historically disrespected inflexions
        disrespected_indices = set()

        # Detect inflexions (need at least 3 candles)
        if n < 3:
            return pd.DataFrame({
                'InflexionType': inflexion_type,
                'Level': level,
                'Respected': respected,
                'StatusIndex': status_index
            })

        # 3-point comparison for concave (peaks) and convex (valleys)
        for i in range(1, n - 1):
            # Skip if already disrespected
            if i in disrespected_indices:
                continue

            # Concave detection (peak)
            if high_prices[i] >= high_prices[i-1] and high_prices[i] > high_prices[i+1]:
                inflexion_type[i] = 1
                level[i] = high_prices[i]

                # Evaluate respect/disrespect status
                future_closes = close_prices[i+1:]
                future_opens = open_prices[i+1:]
                future_highs = high_prices[i+1:]

                # Disrespected if any close or open goes above level
                disrespected_mask = (future_closes > level[i]) | (future_opens > level[i])
                if np.any(disrespected_mask):
                    respected[i] = False
                    status_index[i] = i + 1 + np.argmax(disrespected_mask)
                    disrespected_indices.add(i)
                # Respected if any high reaches level but doesn't close through
                elif np.any(future_highs >= level[i]):
                    respected[i] = True
                    status_index[i] = i + 1 + np.argmax(future_highs >= level[i])
                # Still pending
                else:
                    respected[i] = None
                    status_index[i] = 0

            # Convex detection (valley)
            if low_prices[i] <= low_prices[i-1] and low_prices[i] < low_prices[i+1]:
                inflexion_type[i] = -1
                level[i] = low_prices[i]

                # Evaluate respect/disrespect status
                future_closes = close_prices[i+1:]
                future_opens = open_prices[i+1:]
                future_lows = low_prices[i+1:]

                # Disrespected if any close or open goes below level
                disrespected_mask = (future_closes < level[i]) | (future_opens < level[i])
                if np.any(disrespected_mask):
                    respected[i] = False
                    status_index[i] = i + 1 + np.argmax(disrespected_mask)
                    disrespected_indices.add(i)
                # Respected if any low reaches level but doesn't close through
                elif np.any(future_lows <= level[i]):
                    respected[i] = True
                    status_index[i] = i + 1 + np.argmax(future_lows <= level[i])
                # Still pending
                else:
                    respected[i] = None
                    status_index[i] = 0

        # Convert to Series
        inflexion_type_series = pd.Series(inflexion_type, name="InflexionType")
        level_series = pd.Series(level, name="Level")
        respected_series = pd.Series(respected, name="Respected")
        status_index_series = pd.Series(status_index, name="StatusIndex")

        return pd.concat([inflexion_type_series, level_series, respected_series, status_index_series], axis=1)

    @classmethod
    def bos(cls, ohlc: DataFrame, inflexions: DataFrame, close_break: bool = True) -> DataFrame:
        """
        Break of Structure - Trend-Aware Detection

        Detects when price breaks through trend-defining inflexion levels, signaling
        a potential trend change. More sophisticated than pattern matching as it:
        - Maintains explicit trend state (bullish/bearish/undefined)
        - Distinguishes between trend-defining and recent pivots
        - Requires close/open confirmation (more conservative)

        Logic:
        - Bullish Trend: Higher Highs (HH) and Higher Lows (HL)
        - Bearish Trend: Lower Highs (LH) and Lower Lows (LL)
        - BOS occurs when price closes/opens beyond the last trend-defining pivot

        Parameters:
        inflexions: DataFrame - output from inflexion_points() function
        close_break: bool - if True, requires close/open beyond level (default: True)
                           if False, any price breach counts

        Returns:
        BOS = 1 if bullish break (bearish trend broken),
             -1 if bearish break (bullish trend broken),
             NaN if no BOS
        Level = price level that was broken
        BrokenIndex = index where BOS occurred
        Trend = current trend state: 1 (bullish), -1 (bearish), 0 (undefined)
        TrendStartIndex = index where current trend began

        Example:
        ```python
        inflexions = smc_custom.inflexion_points(df)
        bos_data = smc_custom.bos(df, inflexions, close_break=True)

        # Get current trend
        current_trend = bos_data['Trend'].iloc[-1]
        if current_trend == 1:
            print("Bullish trend")
        elif current_trend == -1:
            print("Bearish trend")
        ```
        """
        n = len(ohlc)

        # Extract price arrays
        high_prices = ohlc["high"].values
        low_prices = ohlc["low"].values
        close_prices = ohlc["close"].values
        open_prices = ohlc["open"].values

        # Extract inflexion data
        inflexion_types = inflexions["InflexionType"].values
        inflexion_levels = inflexions["Level"].values

        # Initialize output arrays
        bos = np.full(n, np.nan, dtype=np.float32)
        bos_level = np.full(n, np.nan, dtype=np.float32)
        broken_index = np.zeros(n, dtype=np.int32)
        trend = np.zeros(n, dtype=np.int32)  # 1=bullish, -1=bearish, 0=undefined
        trend_start_index = np.zeros(n, dtype=np.int32)

        # State tracking
        last_trend_max = None  # (index, price) of highest concave in trend
        last_trend_min = None  # (index, price) of lowest convex in trend
        last_max = None        # (index, price) of most recent concave
        last_min = None        # (index, price) of most recent convex
        current_trend = 0      # Current trend state
        current_trend_start = 0

        # Get inflexion points
        inflexion_indices = np.where(~np.isnan(inflexion_types))[0]

        if len(inflexion_indices) < 2:
            # Not enough inflexions
            return pd.DataFrame({
                'BOS': bos,
                'Level': bos_level,
                'BrokenIndex': broken_index,
                'Trend': trend,
                'TrendStartIndex': trend_start_index
            })

        # Pre-compute most recent inflexions before each index for BOS level calculation
        # For each candle, track the closest previous concave (peak) and convex (valley)
        closest_prev_concave = {}  # {index: (inflexion_index, price)}
        closest_prev_convex = {}   # {index: (inflexion_index, price)}

        last_concave = None
        last_convex = None

        for i in range(n):
            # Store current closest inflexions for this index
            if last_concave is not None:
                closest_prev_concave[i] = last_concave
            if last_convex is not None:
                closest_prev_convex[i] = last_convex

            # Update if there's an inflexion at this index
            if i in inflexion_indices:
                if inflexion_types[i] == 1:  # Concave
                    last_concave = (i, inflexion_levels[i])
                elif inflexion_types[i] == -1:  # Convex
                    last_convex = (i, inflexion_levels[i])

        # Process each candle
        for i in range(n):
            # Check for BOS first (before processing new inflexions)
            if close_break:
                check_price_high = close_prices[i] if close_prices[i] > open_prices[i] else open_prices[i]
                check_price_low = close_prices[i] if close_prices[i] < open_prices[i] else open_prices[i]
            else:
                check_price_high = high_prices[i]
                check_price_low = low_prices[i]

            # Check for bearish trend break (bullish BOS)
            if current_trend == -1 and last_trend_max is not None:
                if check_price_high > last_trend_max[1]:
                    # BOS detected!
                    bos[i] = 1
                    # Level = closest previous convex (valley) before BOS for invalidation
                    if i in closest_prev_convex:
                        bos_level[i] = closest_prev_convex[i][1]
                    elif last_trend_min is not None:
                        bos_level[i] = last_trend_min[1]
                    else:
                        bos_level[i] = last_trend_max[1]
                    broken_index[i] = i
                    current_trend = 1  # Switch to bullish
                    current_trend_start = i
                    if last_min is not None:
                        last_trend_min = last_min
                    if last_max is not None:
                        last_trend_max = last_max

            # Check for bullish trend break (bearish BOS)
            elif current_trend == 1 and last_trend_min is not None:
                if check_price_low < last_trend_min[1]:
                    # BOS detected!
                    bos[i] = -1
                    # Level = closest previous concave (peak) before BOS for invalidation
                    if i in closest_prev_concave:
                        bos_level[i] = closest_prev_concave[i][1]
                    elif last_trend_max is not None:
                        bos_level[i] = last_trend_max[1]
                    else:
                        bos_level[i] = last_trend_min[1]
                    broken_index[i] = i
                    current_trend = -1  # Switch to bearish
                    current_trend_start = i
                    if last_max is not None:
                        last_trend_max = last_max
                    if last_min is not None:
                        last_trend_min = last_min

            # Update trend state
            trend[i] = current_trend
            trend_start_index[i] = current_trend_start

            # Process new inflexion at this candle
            if i not in inflexion_indices:
                continue

            inflx_type = inflexion_types[i]
            inflx_price = inflexion_levels[i]

            # Update last_max and last_min
            if inflx_type == 1:  # Concave
                last_max = (i, inflx_price)
            elif inflx_type == -1:  # Convex
                last_min = (i, inflx_price)

            # Phase 1: Seed first pivots
            if last_trend_max is None and inflx_type == 1:
                last_trend_max = (i, inflx_price)
                continue
            if last_trend_min is None and inflx_type == -1:
                last_trend_min = (i, inflx_price)
                continue

            # Phase 2: Establish initial trend
            if current_trend == 0:
                if last_trend_max is not None and inflx_type == 1:
                    if inflx_price < last_trend_max[1]:
                        current_trend = -1  # Bearish
                        current_trend_start = last_trend_max[0]
                    elif inflx_price > last_trend_max[1]:
                        current_trend = 1   # Bullish
                        current_trend_start = last_trend_min[0] if last_trend_min else i
                elif last_trend_min is not None and inflx_type == -1:
                    if inflx_price > last_trend_min[1]:
                        current_trend = 1   # Bullish
                        current_trend_start = last_trend_min[0]
                    elif inflx_price < last_trend_min[1]:
                        current_trend = -1  # Bearish
                        current_trend_start = last_trend_max[0] if last_trend_max else i

                # Update arrays
                if current_trend != 0:
                    trend[i] = current_trend
                    trend_start_index[i] = current_trend_start
                continue

            # Phase 3: Trend maintenance
            if current_trend == -1:  # Bearish trend
                if inflx_type == 1 and inflx_price < last_trend_max[1]:
                    # Lower high - update trend max
                    last_trend_max = (i, inflx_price)
                elif inflx_type == -1 and inflx_price < last_trend_min[1]:
                    # Lower low - update trend min
                    last_trend_min = (i, inflx_price)

            elif current_trend == 1:  # Bullish trend
                if inflx_type == 1 and inflx_price > last_trend_max[1]:
                    # Higher high - update trend max
                    last_trend_max = (i, inflx_price)
                elif inflx_type == -1 and inflx_price > last_trend_min[1]:
                    # Higher low - update trend min
                    last_trend_min = (i, inflx_price)

        # Replace 0s with NaN for BrokenIndex where no BOS
        broken_index = np.where(np.isnan(bos), np.nan, broken_index)

        # Convert to Series
        bos_series = pd.Series(bos, name="BOS")
        level_series = pd.Series(bos_level, name="Level")
        broken_series = pd.Series(broken_index, name="BrokenIndex")
        trend_series = pd.Series(trend, name="Trend")
        trend_start_series = pd.Series(trend_start_index, name="TrendStartIndex")

        return pd.concat([bos_series, level_series, broken_series, trend_series, trend_start_series], axis=1)

    @classmethod
    def ob(cls, ohlc: DataFrame, bos: DataFrame, inflexions: DataFrame) -> DataFrame:
        """
        Order Block - Detection based on broken inflection levels.

        For each BOS, finds the inflection point that was broken and creates
        an OB zone spanning from that inflection to the BOS.

        Definition:
        - Bearish OB (at bearish BOS, trend bullish→bearish):
          - Zone from broken HIGH inflection to BOS
          - Top = Level (the broken level)
          - Bottom = min price in range

        - Bullish OB (at bullish BOS, trend bearish→bullish):
          - Zone from broken LOW inflection to BOS
          - Bottom = Level (the broken level)
          - Top = max price in range

        Parameters:
        ohlc: DataFrame - OHLCV data
        bos: DataFrame - output from bos() (contains BOS, Level columns)
        inflexions: DataFrame - output from inflexion_points()

        Returns:
        OB = 1 if bullish OB, -1 if bearish OB, NaN if none
        Top = top of the order block zone
        Bottom = bottom of the order block zone
        StartIndex = index of the broken inflection point
        EndIndex = index of the BOS

        Example:
        ```python
        inflexions = smc_custom.inflexion_points(df)
        bos = smc_custom.bos(df, inflexions)
        ob_data = smc_custom.ob(df, bos, inflexions)

        # Get all bullish OBs
        bullish_obs = ob_data[ob_data['OB'] == 1]
        # Get all bearish OBs
        bearish_obs = ob_data[ob_data['OB'] == -1]
        ```
        """
        n = len(ohlc)

        # Extract price arrays
        high_prices = ohlc["high"].values
        low_prices = ohlc["low"].values

        # Extract BOS data
        bos_values = bos["BOS"].values

        # Extract inflexion data
        inflexion_types = inflexions["InflexionType"].values
        inflexion_levels = inflexions["Level"].values

        # Initialize output arrays
        ob = np.full(n, np.nan, dtype=np.float32)
        top_arr = np.full(n, np.nan, dtype=np.float32)
        bottom_arr = np.full(n, np.nan, dtype=np.float32)
        start_idx_arr = np.full(n, np.nan, dtype=np.float32)
        end_idx_arr = np.full(n, np.nan, dtype=np.float32)

        # Process each BOS event
        for i in range(n):
            if np.isnan(bos_values[i]):
                continue

            bos_type = bos_values[i]
            bos_idx = i

            # Simple approach: find the most recent relevant inflection
            # - Bullish BOS broke a peak (type 1) → find most recent peak
            # - Bearish BOS broke a valley (type -1) → find most recent valley
            inflexion_idx = None
            target_type = 1 if bos_type == 1 else -1  # Bullish BOS broke peak, Bearish broke valley

            for j in range(i - 1, -1, -1):
                if np.isnan(inflexion_types[j]):
                    continue
                if inflexion_types[j] == target_type:
                    inflexion_idx = j
                    break

            if inflexion_idx is None:
                continue

            # Calculate OB zone from inflection to BOS
            start_idx = inflexion_idx
            end_idx = bos_idx
            broken_level = inflexion_levels[inflexion_idx]

            if bos_type == 1:  # Bullish BOS → Bullish OB
                # Broke above a peak - OB from peak level down to lowest low in range
                ob[start_idx] = 1
                top_arr[start_idx] = broken_level
                bottom_arr[start_idx] = min(low_prices[start_idx:end_idx + 1])
                start_idx_arr[start_idx] = start_idx
                end_idx_arr[start_idx] = end_idx

            elif bos_type == -1:  # Bearish BOS → Bearish OB
                # Broke below a valley - OB from highest high down to valley level
                ob[start_idx] = -1
                top_arr[start_idx] = max(high_prices[start_idx:end_idx + 1])
                bottom_arr[start_idx] = broken_level
                start_idx_arr[start_idx] = start_idx
                end_idx_arr[start_idx] = end_idx

        # Convert to Series
        ob_series = pd.Series(ob, name="OB")
        top_series = pd.Series(top_arr, name="Top")
        bottom_series = pd.Series(bottom_arr, name="Bottom")
        start_idx_series = pd.Series(start_idx_arr, name="StartIndex")
        end_idx_series = pd.Series(end_idx_arr, name="EndIndex")

        return pd.concat([ob_series, top_series, bottom_series, start_idx_series, end_idx_series], axis=1)
