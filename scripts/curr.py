"""
GMM-Based Price Distribution Detection with Normality Verification

Detects price distribution regimes using Gaussian Mixture Models (GMM).
Identifies which distribution the current price belongs to and verifies
that the distribution is actually normal.

Algorithm:
1. Use GMM to find how many distributions exist
2. Identify which component contains the current price
3. Extract price levels belonging to that component
4. Test if those levels are normally distributed
5. Keep expanding window until normality is confirmed

Usage:
    python scripts/curr.py --symbol TSLA --timeframe 1h --step 0.1

Copy-paste friendly: All code can be copied into a Jupyter notebook cell.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Add parent directory for imports (when running as script)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import DataLoader


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_price_levels(df: pd.DataFrame, step: float = None,
                          max_points_per_candle: int = 50) -> np.ndarray:
    """
    Generate discrete price levels from candle high-low ranges.

    For each candle, generates price points from low to high.
    Step is auto-calculated if not provided based on price range.

    Args:
        df: DataFrame with 'high' and 'low' columns
        step: Price increment (auto-calculated if None)
        max_points_per_candle: Maximum points per candle (default 50)

    Returns:
        Array of all price levels
    """
    lows = df['low'].values
    highs = df['high'].values

    # Auto-calculate step based on average candle range
    if step is None:
        avg_range = np.mean(highs - lows)
        step = max(avg_range / max_points_per_candle, 0.01)

    # Calculate number of points per candle, capped
    n_points = np.minimum(
        np.ceil((highs - lows) / step + 1).astype(int),
        max_points_per_candle
    )
    total_points = n_points.sum()

    # Pre-allocate array
    all_prices = np.empty(total_points)

    idx = 0
    for low, high, n in zip(lows, highs, n_points):
        all_prices[idx:idx + n] = np.linspace(low, high, n)
        idx += n

    return all_prices


# =============================================================================
# NORMALITY TESTING FUNCTIONS
# =============================================================================

def run_normality_tests(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run multiple normality tests on the data.

    Tests: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, Jarque-Bera

    Args:
        data: Array of values to test
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with test results
    """
    results = {}

    # Need minimum data points
    if len(data) < 20:
        return {
            'shapiro_wilk': {'is_normal': None, 'note': 'Insufficient data'},
            'dagostino_pearson': {'is_normal': None, 'note': 'Insufficient data'},
            'anderson_darling': {'is_normal': None, 'note': 'Insufficient data'},
            'jarque_bera': {'is_normal': None, 'note': 'Insufficient data'},
        }

    # Shapiro-Wilk test (sample if data is too large)
    if len(data) > 5000:
        sample_data = np.random.choice(data, size=5000, replace=False)
        shapiro_note = " (sampled 5000 points)"
    else:
        sample_data = data
        shapiro_note = ""

    stat, p = stats.shapiro(sample_data)
    results['shapiro_wilk'] = {
        'statistic': stat,
        'p_value': p,
        'is_normal': p > alpha,
        'note': shapiro_note
    }

    # D'Agostino-Pearson test
    stat, p = stats.normaltest(data)
    results['dagostino_pearson'] = {
        'statistic': stat,
        'p_value': p,
        'is_normal': p > alpha
    }

    # Anderson-Darling test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = stats.anderson(data, dist='norm')
    critical_5pct = result.critical_values[2]
    results['anderson_darling'] = {
        'statistic': result.statistic,
        'critical_value_5pct': critical_5pct,
        'is_normal': result.statistic < critical_5pct
    }

    # Jarque-Bera test
    stat, p = stats.jarque_bera(data)
    results['jarque_bera'] = {
        'statistic': stat,
        'p_value': p,
        'is_normal': p > alpha
    }

    return results


def is_distribution_normal(test_results: dict, min_passing: int = 2) -> tuple:
    """
    Determine if the data is normally distributed based on test results.

    Args:
        test_results: Dictionary from run_normality_tests()
        min_passing: Minimum number of tests that must pass (default 2)

    Returns:
        Tuple of (is_normal: bool, tests_passed: int, total_tests: int)
    """
    passed = sum(1 for t in test_results.values() if t.get('is_normal') is True)
    total = sum(1 for t in test_results.values() if t.get('is_normal') is not None)
    return passed >= min_passing, passed, total


# =============================================================================
# GMM FUNCTIONS
# =============================================================================

def find_optimal_components(data: np.ndarray, max_components: int = 3) -> tuple:
    """
    Find optimal number of GMM components using BIC.

    Returns:
        Tuple of (best_n_components, bic_scores, best_gmm_model)
    """
    data = data.reshape(-1, 1)
    bics = []
    models = []

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        models.append(gmm)

    best_n = np.argmin(bics) + 1
    return best_n, bics, models[best_n - 1]


def get_component_prices(price_levels: np.ndarray, gmm, component: int) -> np.ndarray:
    """
    Extract price levels that belong to a specific GMM component.

    Uses soft assignment (probability > 0.5) to assign prices to components.

    Args:
        price_levels: Array of all price levels
        gmm: Fitted GMM model
        component: Component index to extract

    Returns:
        Array of price levels belonging to this component
    """
    data = price_levels.reshape(-1, 1)
    probas = gmm.predict_proba(data)
    mask = probas[:, component] > 0.5
    return price_levels[mask]


def get_price_distribution(price_levels: np.ndarray, current_price: float,
                           n_components: int = None) -> dict:
    """
    Fit GMM and identify which distribution the current price belongs to.
    Also extracts prices from the current price's component for normality testing.
    """
    data = price_levels.reshape(-1, 1)

    # Auto-detect components if not specified
    if n_components is None:
        n_components, bics, gmm = find_optimal_components(data)
    else:
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=3)
        gmm.fit(data)
        bics = None

    # Classify current price
    component = gmm.predict([[current_price]])[0]
    proba = gmm.predict_proba([[current_price]])[0]

    # Get component parameters
    mean = gmm.means_[component][0]
    std = np.sqrt(gmm.covariances_[component][0][0])

    # Extract prices belonging to this component
    component_prices = get_component_prices(price_levels, gmm, component)

    return {
        'n_components': n_components,
        'current_component': component,
        'current_mean': mean,
        'current_std': std,
        'confidence': proba[component],
        'all_means': gmm.means_.flatten(),
        'all_stds': np.sqrt(gmm.covariances_.flatten()),
        'weights': gmm.weights_,
        'gmm': gmm,
        'bics': bics,
        'component_prices': component_prices  # Prices for normality testing
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_gmm_distributions(price_levels: np.ndarray, result: dict,
                           current_price: float, normality_results: dict = None,
                           save_path: str = None):
    """
    Visualize GMM components and highlight current price's distribution.
    Shows normality test results if provided.

    Args:
        price_levels: Array of price levels
        result: Dictionary from get_price_distribution()
        current_price: Current price value
        normality_results: Optional normality test results
        save_path: If provided, save figure to this path instead of showing
    """
    n_plots = 3 if normality_results else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # Left: Histogram with GMM components
    axes[0].hist(price_levels, bins=100, density=True, alpha=0.5,
                 color='gray', label='All Data')

    x = np.linspace(price_levels.min(), price_levels.max(), 1000)

    colors = ['blue', 'red', 'green']
    for i in range(result['n_components']):
        mean = result['all_means'][i]
        std = result['all_stds'][i]
        weight = result['weights'][i]

        component_pdf = weight * stats.norm.pdf(x, mean, std)
        label = f'Comp {i}: mu={mean:.1f}, std={std:.1f}'

        if i == result['current_component']:
            label += ' <- CURRENT'
            axes[0].plot(x, component_pdf, color=colors[i % len(colors)],
                        linewidth=3, label=label)
        else:
            axes[0].plot(x, component_pdf, color=colors[i % len(colors)],
                        linewidth=1.5, linestyle='--', label=label)

    axes[0].axvline(current_price, color='black', linestyle=':', linewidth=2,
                    label=f'Current: {current_price:.1f}')
    axes[0].set_xlabel('Price Level')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'GMM: {result["n_components"]} Component(s) Detected')
    axes[0].legend(fontsize=8)

    # Middle: BIC comparison
    if result['bics'] is not None:
        n_range = range(1, len(result['bics']) + 1)
        colors_bar = ['green' if i == result['n_components'] else 'gray' for i in n_range]
        axes[1].bar(n_range, result['bics'], color=colors_bar)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('BIC (lower is better)')
        axes[1].set_title('Model Selection via BIC')
        axes[1].set_xticks(list(n_range))
    else:
        axes[1].text(0.5, 0.5, 'BIC not computed',
                     ha='center', va='center', transform=axes[1].transAxes)

    # Right: Normality test results (if provided)
    if normality_results and n_plots == 3:
        # Q-Q plot for component prices
        component_prices = result['component_prices']
        if len(component_prices) >= 20:
            stats.probplot(component_prices, dist="norm", plot=axes[2])

            # Add normality verdict
            is_normal, passed, total = is_distribution_normal(normality_results)
            verdict = "NORMAL" if is_normal else "NOT NORMAL"
            color = 'green' if is_normal else 'red'
            axes[2].set_title(f'Q-Q Plot: {verdict} ({passed}/{total} tests passed)',
                             color=color)
        else:
            axes[2].text(0.5, 0.5, f'Insufficient data\n({len(component_prices)} points)',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Q-Q Plot')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def print_normality_results(test_results: dict):
    """Print formatted normality test results."""
    print("\nNormality Test Results:")
    print("-" * 40)
    for name, result in test_results.items():
        if result.get('is_normal') is None:
            status = "N/A"
        else:
            status = "PASS" if result['is_normal'] else "FAIL"

        if 'p_value' in result and result['p_value'] is not None:
            print(f"  {name}: {status} (p={result['p_value']:.4f})")
        elif 'statistic' in result and result['statistic'] is not None:
            print(f"  {name}: {status} (stat={result['statistic']:.4f})")
        else:
            print(f"  {name}: {result.get('note', 'N/A')}")
    print("-" * 40)


# =============================================================================
# MAIN DETECTION LOOP
# =============================================================================

def run_gmm_detection(symbol: str = 'TSLA', timeframe: str = '1h',
                      step: float = None, initial_window: int = 30,
                      max_iterations: int = 200, window_increment: int = 5,
                      min_tests_passing: int = 2, alpha: float = 0.05,
                      confidence_threshold: float = 0.9,
                      require_normality: bool = False,
                      verbose: bool = True):
    """
    Run GMM distribution detection loop.

    The algorithm:
    1. Start with initial_window candles
    2. Use GMM to find distribution components
    3. Identify which component contains the current price
    4. Extract price levels from that component
    5. Run normality tests (informational)
    6. Stop when confidence is high enough OR normality achieved

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        step: Price level increment (auto-calculated if None)
        initial_window: Starting window size (candles)
        max_iterations: Maximum iterations before stopping
        window_increment: Candles to add each iteration
        min_tests_passing: Minimum normality tests to consider "normal"
        alpha: Significance level for normality tests
        confidence_threshold: Stop when GMM confidence exceeds this (default 0.9)
        require_normality: If True, only stop when normality achieved (default False)
        verbose: Print progress (default True)

    Returns:
        Tuple of (result_dict, normality_results, current_price, final_window_size)
        or None if data loading fails
    """
    # Load data
    if verbose:
        print(f"Loading {symbol} {timeframe} data...")
    loader = DataLoader(symbol=symbol)
    df = loader.get_data(timeframe=timeframe)

    if df is None or len(df) == 0:
        print(f"Error: No data found for {symbol} {timeframe}")
        return None

    if verbose:
        print(f"Loaded {len(df)} candles")

    current_price = df['close'].iloc[-1]
    if verbose:
        print(f"Current price: {current_price:.2f}")
        mode = "normality required" if require_normality else f"confidence > {confidence_threshold}"
        print(f"Searching for distribution regime (stop when: {mode})...\n")

    counter = 0
    window_size = initial_window
    result = None
    normality_results = None
    passed = 0
    total = 4

    while counter < max_iterations and window_size <= len(df):
        # Get subset of most recent candles
        subset = df.tail(window_size)
        price_levels = generate_price_levels(subset, step=step)

        # Need minimum data for meaningful analysis
        if len(price_levels) < 100:
            counter += 1
            window_size += window_increment
            continue

        # Fit GMM and get component containing current price
        result = get_price_distribution(price_levels, current_price)

        # Get prices from current price's component
        component_prices = result['component_prices']

        # Need enough component prices for normality tests
        if len(component_prices) < 20:
            if verbose and counter % 10 == 0:
                print(f"Iter {counter}: Window={window_size}, "
                      f"component has only {len(component_prices)} points (need 20+)")
            counter += 1
            window_size += window_increment
            continue

        # Run normality tests on the component prices
        normality_results = run_normality_tests(component_prices, alpha=alpha)
        is_normal, passed, total = is_distribution_normal(normality_results, min_tests_passing)

        # Determine stopping criteria
        confident = result['confidence'] > confidence_threshold
        should_stop = is_normal if require_normality else (confident or is_normal)

        if should_stop:
            if verbose:
                status = "NORMAL" if is_normal else f"HIGH CONFIDENCE ({result['confidence']:.1%})"
                print(f"\nFound distribution at iteration {counter}! ({status})")
                print(f"  Window: {window_size} candles")
                print(f"  Components detected: {result['n_components']}")
                print(f"  Current price ({current_price:.2f}) in component {result['current_component']}")
                print(f"  Component mean: {result['current_mean']:.2f}")
                print(f"  Component std:  {result['current_std']:.2f}")
                print(f"  Price range: [{result['current_mean'] - 2*result['current_std']:.2f}, "
                      f"{result['current_mean'] + 2*result['current_std']:.2f}] (2 std)")
                print(f"  Component prices: {len(component_prices)} points")
                print(f"  GMM confidence: {result['confidence']:.1%}")
                print(f"  Normality: {passed}/{total} tests passed")

            return result, normality_results, current_price, window_size

        # Progress logging every 5 iterations
        if verbose and counter % 5 == 0:
            print(f"Iter {counter}: Window={window_size}, "
                  f"{result['n_components']} comp, "
                  f"conf={result['confidence']:.1%}, "
                  f"norm={passed}/{total}")

        counter += 1
        window_size += window_increment

    # Max iterations or data exhausted - return best result
    if verbose:
        print(f"\nSearch stopped at iteration {counter}")
        print(f"Final window: {window_size} candles")
        if result:
            print(f"Best result: {result['n_components']} components, "
                  f"confidence={result['confidence']:.1%}, "
                  f"normality={passed}/{total}")

    return result, normality_results, current_price, window_size


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='GMM-based price distribution detection with normality verification'
    )
    parser.add_argument('--symbol', type=str, default='TSLA', help='Stock symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Data timeframe')
    parser.add_argument('--step', type=float, default=None, help='Price level step (auto if not set)')
    parser.add_argument('--initial-window', type=int, default=30, help='Initial window size')
    parser.add_argument('--max-iterations', type=int, default=200, help='Max iterations')
    parser.add_argument('--min-tests', type=int, default=2, help='Min normality tests to pass')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--confidence', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--require-normality', action='store_true',
                        help='Only stop when normality tests pass')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save plot to file instead of displaying')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')

    args = parser.parse_args()

    # Run detection
    output = run_gmm_detection(
        symbol=args.symbol,
        timeframe=args.timeframe,
        step=args.step,
        initial_window=args.initial_window,
        max_iterations=args.max_iterations,
        min_tests_passing=args.min_tests,
        alpha=args.alpha,
        confidence_threshold=args.confidence,
        require_normality=args.require_normality
    )

    if output is not None:
        result, normality_results, current_price, window_size = output

        print(f"\nFinal window: {window_size} candles")
        if normality_results:
            print_normality_results(normality_results)

        if not args.no_plot:
            print("\nGenerating visualization...")
            price_levels = generate_price_levels(
                DataLoader(symbol=args.symbol).get_data(timeframe=args.timeframe).tail(window_size),
                step=args.step
            )
            plot_gmm_distributions(price_levels, result, current_price,
                                   normality_results, save_path=args.save_plot)
