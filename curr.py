from src.data_loader import DataLoader
symbol = 'META'
loader = DataLoader(symbol=symbol)
high_tf = '1h'
mid_tf = '5min'
low_tf = '1min'

df_high = loader.get_data(high_tf, force_refresh=False)
df_mid = loader.get_data(mid_tf, force_refresh=False)
df_low = loader.get_data(low_tf, force_refresh=False)

high_prices = list(df_high.close)
mid_prices = list(df_mid.close)
low_prices = list(df_low.close)

mid_prices_focus = mid_prices[-500:]  # Focus on the last 500 data points

from src.arma_trend_analysis import ARMA


arma = ARMA()


# Fit ARMA
result = arma.fit_arma(mid_prices_focus, auto_select=True, max_order=3)

# Print summary
arma.print_summary(result)

# Forecast next 5 periods
forecast = arma.forecast_returns(result, steps=5)
print("\nFORECAST (next 5 periods):")
print(forecast['forecast'])




import matplotlib.pyplot as plt
print('ar_coeffs -->', result.get('ar_coeffs'))
print('ma_coeffs -->', result.get('ma_coeffs'))
plt.hist(result.get('residuals'))
plt.show()