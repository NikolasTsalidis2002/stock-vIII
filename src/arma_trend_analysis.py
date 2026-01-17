import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import warnings

warnings.filterwarnings('ignore')

class ARMA:
    @staticmethod
    def fit_arma(prices, p=1, q=1, auto_select=True, max_order=5):
        """
        Fit an ARMA model to asset prices.

        Parameters
        ----------
        prices : array-like
            Original asset prices (not returns).
        p : int
            AR order (used if auto_select=False).
        q : int
            MA order (used if auto_select=False).
        auto_select : bool
            If True, automatically select best (p, q) using AIC.
        max_order : int
            Maximum order to search when auto_select=True.

        Returns
        -------
        dict with:
            - 'drift': constant/mean term (trend direction indicator)
            - 'ar_coeffs': AR coefficients
            - 'ma_coeffs': MA coefficients
            - 'sigma2': variance of residuals
            - 'fitted_returns': fitted values for log returns
            - 'residuals': model residuals
            - 'aic': AIC of the model
            - 'bic': BIC of the model
            - 'order': (p, q) used
            - 'log_returns': the log returns series
            - 'trend_signal': 'bullish', 'bearish', or 'neutral'
        """
        prices = np.asarray(prices).flatten()

        # Convert to log returns
        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)

        if auto_select:
            p, q = ARMA._select_order(log_returns, max_order)

        # Fit ARIMA(p, 0, q) on returns = ARMA(p, q)
        # Using ARIMA with d=0 is equivalent to ARMA
        model = ARIMA(log_returns, order=(p, 0, q), trend='c')
        fitted = model.fit()

        # Extract components - create a dict from param_names and params
        param_names = fitted.param_names
        param_values = fitted.params
        params = dict(zip(param_names, param_values))

        drift = params.get('const', 0)
        ar_coeffs = [params.get(f'ar.L{i}', 0) for i in range(1, p + 1)]
        ma_coeffs = [params.get(f'ma.L{i}', 0) for i in range(1, q + 1)]

        # Calculate residual volatility and SNR (Signal-to-Noise Ratio)
        residual_std = np.sqrt(np.var(fitted.resid))
        snr = abs(drift) / residual_std if residual_std > 0 else 0

        # Determine trend signal - SNR must be >= 0.05 for meaningful trend
        # Per Lo & MacKinlay (1988) and Tsay (2010), thresholds around 0.05-0.1
        # indicate practical significance
        if snr < 0.05:
            trend_signal = 'neutral'
        elif drift > 0:
            trend_signal = 'bullish'
        elif drift < 0:
            trend_signal = 'bearish'
        else:
            trend_signal = 'neutral'

        return {
            'drift': drift,
            'ar_coeffs': ar_coeffs,
            'ma_coeffs': ma_coeffs,
            'sigma2': np.var(fitted.resid),
            'snr': snr,
            'fitted_returns': fitted.fittedvalues,
            'residuals': fitted.resid,
            'aic': fitted.aic,
            'bic': fitted.bic,
            'order': (p, q),
            'log_returns': log_returns,
            'trend_signal': trend_signal,
            'model': fitted  # full model object for further analysis
        }


    @staticmethod
    def _select_order(returns, max_order):
        """Select best (p, q) using AIC."""
        best_aic = np.inf
        best_order = (1, 1)

        for p in range(0, max_order + 1):
            for q in range(0, max_order + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(returns, order=(p, 0, q), trend='c')
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, q)
                except:
                    continue

        return best_order


    @staticmethod
    def forecast_returns(result, steps=10):
        """
        Forecast future log returns.

        Parameters
        ----------
        result : dict
            Output from fit_arma().
        steps : int
            Number of periods to forecast.

        Returns
        -------
        dict with:
            - 'forecast': point forecasts
            - 'conf_int': confidence intervals (95%)
        """
        forecast_obj = result['model'].get_forecast(steps=steps)

        return {
            'forecast': forecast_obj.predicted_mean,
            'conf_int': forecast_obj.conf_int(alpha=0.05)
        }


    @staticmethod
    def print_summary(result):
        """Print a readable summary of the ARMA results."""
        print("=" * 50)
        print("ARMA MODEL SUMMARY")
        print("=" * 50)
        print(f"Order: ARMA({result['order'][0]}, {result['order'][1]})")
        print(f"AIC: {result['aic']:.4f}")
        print(f"BIC: {result['bic']:.4f}")
        print("-" * 50)
        print("COEFFICIENTS:")
        print(f"  Drift (μ): {result['drift']:.6f}")

        if result['ar_coeffs']:
            print("  AR coefficients:")
            for i, coef in enumerate(result['ar_coeffs'], 1):
                print(f"    φ_{i}: {coef:.6f}")

        if result['ma_coeffs']:
            print("  MA coefficients:")
            for i, coef in enumerate(result['ma_coeffs'], 1):
                print(f"    θ_{i}: {coef:.6f}")

        print(f"  Residual variance (σ²): {result['sigma2']:.6f}")
        print(f"  Residual std (σ): {np.sqrt(result['sigma2']):.6f}")
        print("-" * 50)
        print("SIGNAL-TO-NOISE RATIO:")
        print(f"  SNR: {result['snr']:.4f}")
        print(f"  Threshold: 0.05 (trend meaningful if SNR >= 0.05)")
        snr_status = "✓ Above threshold" if result['snr'] >= 0.05 else "✗ Below threshold"
        print(f"  Status: {snr_status}")
        print("-" * 50)
        print(f"TREND SIGNAL: {result['trend_signal'].upper()}")

        if result['trend_signal'] == 'bullish':
            print("  → Positive drift with sufficient SNR indicates upward trend")
        elif result['trend_signal'] == 'bearish':
            print("  → Negative drift with sufficient SNR indicates downward trend")
        else:
            if result['snr'] < 0.05:
                print("  → SNR below threshold - trend not statistically meaningful")
            else:
                print("  → Drift near zero, no clear directional trend")
        print("=" * 50)


# # Example usage
# if __name__ == "__main__":
#     # Generate sample data (random walk with positive drift = bullish)
#     np.random.seed(42)
#     n = 500
#     drift = 0.001  # positive drift
#     returns = drift + 0.02 * np.random.randn(n)
#     prices = 100 * np.exp(np.cumsum(returns))  # convert to prices

#     # Fit ARMA
#     result = fit_arma(prices, auto_select=True, max_order=3)

#     # Print summary
#     print_summary(result)

#     # Forecast next 5 periods
#     forecast = forecast_returns(result, steps=5)
#     print("\nFORECAST (next 5 periods):")
#     print(forecast['forecast'])
