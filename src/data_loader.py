"""
Stock Data Loader
Fetches and caches stock data from Twelve Data API
Supports any stock symbol (default: TSLA)
"""

import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path


class DataLoader:
    """
    Handles downloading and caching stock data from Twelve Data API.

    Attributes:
        api_key (str): Twelve Data API key
        base_url (str): API endpoint
        data_dir (Path): Directory for cached data
        symbol (str): Stock symbol (default: TSLA)
        timezone (str): Timezone for data (default: Europe/Madrid)
    """

    def __init__(self, symbol='TSLA', api_key=None, data_dir=None):
        """
        Initialize the data loader.

        Args:
            symbol (str, optional): Stock symbol to load data for. Default: TSLA
            api_key (str, optional): Twelve Data API key. If None, uses default.
            data_dir (str, optional): Directory for cached data. If None, uses ../data/{symbol}
        """
        # API Configuration
        self.api_key = api_key or '05bd1cfd57be431f89dfca9dd12f2cd2'
        self.base_url = 'https://api.twelvedata.com/time_series'

        # Data Configuration
        self.symbol = symbol.upper()
        self.timezone = 'Europe/Madrid'

        # Set up data directory (use lowercase symbol for directory name)
        if data_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            self.data_dir = project_root / 'data' / self.symbol.lower()
        else:
            self.data_dir = Path(data_dir)

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Supported timeframes
        self.supported_timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1day']

    def fetch_data(self, timeframe='15min', outputsize=5000):
        """
        Fetch TSLA data from Twelve Data API.

        Args:
            timeframe (str): Time interval (1min, 5min, 15min, 30min, 1h, 4h, 1day)
            outputsize (int): Number of candles to fetch (max 5000)

        Returns:
            pd.DataFrame: OHLCV data with columns: time, open, high, low, close, volume,
                         time_interval, name

        Raises:
            Exception: If API request fails or data is invalid
        """
        print(f"📥 Fetching {timeframe} data for {self.symbol}...")

        # Validate timeframe
        if timeframe not in self.supported_timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. "
                           f"Supported: {', '.join(self.supported_timeframes)}")

        # API request parameters
        params = {
            'symbol': self.symbol,
            'interval': timeframe,
            'outputsize': outputsize,
            'apikey': self.api_key,
            'timezone': self.timezone
        }

        # Make API request
        response = requests.get(self.base_url, params=params)

        # Check response
        if response.status_code != 200:
            raise Exception(f"❌ Failed to fetch data. Status code: {response.status_code}, "
                          f"Response: {response.text}")

        data = response.json()

        # Check for API errors
        if 'status' in data and data['status'] == 'error':
            raise Exception(f"❌ API Error: {data.get('message', 'Unknown error')}")

        # Parse data
        try:
            df = pd.DataFrame(data['values'])
        except Exception as e:
            raise Exception(f'❌ Error parsing API response: {e}\nData:\n{data}')

        # Convert columns to proper types
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        df[cols_to_convert] = df[cols_to_convert].astype(float)

        # Rename datetime column to time
        df = df.rename(columns={'datetime': 'time'})
        df['time'] = pd.to_datetime(df['time'])

        # Reverse dataframe so latest data is at the end
        df = df.iloc[::-1].reset_index(drop=True)

        # Add metadata columns
        df['time_interval'] = timeframe
        df['name'] = self.symbol

        print(f"✅ Fetched {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

        return df

    def save_cache(self, df, timeframe):
        """
        Save DataFrame to local cache.

        Args:
            df (pd.DataFrame): Data to save
            timeframe (str): Timeframe identifier for filename
        """
        cache_file = self.data_dir / f"{self.symbol.lower()}_{timeframe}.csv"
        df.to_csv(cache_file, index=False)
        print(f"💾 Cached data to {cache_file}")

    def load_cache(self, timeframe):
        """
        Load DataFrame from local cache.

        Args:
            timeframe (str): Timeframe identifier

        Returns:
            pd.DataFrame or None: Cached data if exists, None otherwise
        """
        cache_file = self.data_dir / f"{self.symbol.lower()}_{timeframe}.csv"

        if not cache_file.exists():
            return None

        try:
            df = pd.read_csv(cache_file)
            df['time'] = pd.to_datetime(df['time'])
            print(f"📂 Loaded {len(df)} candles from cache: {cache_file}")
            return df
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            return None

    def get_cache_info(self, timeframe):
        """
        Get information about cached data.

        Args:
            timeframe (str): Timeframe identifier

        Returns:
            dict: Cache info with keys: exists, file_path, num_candles, start_time, end_time
        """
        cache_file = self.data_dir / f"{self.symbol.lower()}_{timeframe}.csv"

        info = {
            'exists': cache_file.exists(),
            'file_path': str(cache_file),
            'num_candles': 0,
            'start_time': None,
            'end_time': None
        }

        if info['exists']:
            df = self.load_cache(timeframe)
            if df is not None and len(df) > 0:
                info['num_candles'] = len(df)
                info['start_time'] = df['time'].iloc[0]
                info['end_time'] = df['time'].iloc[-1]

        return info

    def get_data(self, timeframe='15min', force_refresh=False):
        """
        Get TSLA data - from cache if available, otherwise fetch from API.

        Args:
            timeframe (str): Time interval
            force_refresh (bool): If True, fetch from API even if cache exists

        Returns:
            pd.DataFrame: OHLCV data
        """
        # Check cache first
        if not force_refresh:
            cached_df = self.load_cache(timeframe)
            if cached_df is not None:
                return cached_df

        # Fetch from API
        df = self.fetch_data(timeframe)

        # Save to cache
        self.save_cache(df, timeframe)

        return df

    def download_all_timeframes(self, timeframes=None, force_refresh=False):
        """
        Download and cache data for multiple timeframes.

        Args:
            timeframes (list, optional): List of timeframes to download.
                                        If None, downloads all supported timeframes.
            force_refresh (bool): If True, fetch from API even if cache exists

        Returns:
            dict: Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = self.supported_timeframes

        results = {}

        print(f"\n{'='*60}")
        print(f"📊 Downloading {len(timeframes)} timeframes for {self.symbol}")
        print(f"{'='*60}\n")

        for i, timeframe in enumerate(timeframes, 1):
            print(f"[{i}/{len(timeframes)}] Processing {timeframe}...")
            try:
                df = self.get_data(timeframe, force_refresh=force_refresh)
                results[timeframe] = df
                print()
            except Exception as e:
                print(f"❌ Error processing {timeframe}: {e}\n")
                results[timeframe] = None

        print(f"{'='*60}")
        print(f"✅ Completed! Successfully downloaded {sum(1 for v in results.values() if v is not None)}/{len(timeframes)} timeframes")
        print(f"{'='*60}\n")

        return results

    def update_cache(self, timeframe):
        """
        Update cached data by fetching latest candles.
        Note: Twelve Data API doesn't support fetching from specific date,
        so this will fetch the latest 5000 candles and merge with existing.

        Args:
            timeframe (str): Timeframe to update

        Returns:
            pd.DataFrame: Updated data
        """
        # Load existing cache
        cached_df = self.load_cache(timeframe)

        if cached_df is None:
            print(f"⚠️ No cache found for {timeframe}, fetching fresh data...")
            return self.get_data(timeframe)

        print(f"🔄 Updating cache for {timeframe}...")
        print(f"   Existing data: {len(cached_df)} candles (until {cached_df['time'].iloc[-1]})")

        # Fetch latest data
        new_df = self.fetch_data(timeframe)

        # Find where to merge (avoid duplicates)
        last_cached_time = cached_df['time'].iloc[-1]
        new_rows = new_df[new_df['time'] > last_cached_time]

        if len(new_rows) == 0:
            print(f"✅ Cache is already up to date!")
            return cached_df

        # Merge
        updated_df = pd.concat([cached_df, new_rows], ignore_index=True)

        # Save updated cache
        self.save_cache(updated_df, timeframe)

        print(f"✅ Added {len(new_rows)} new candles. Total: {len(updated_df)} candles")

        return updated_df

    def validate_data(self, df, timeframe):
        """
        Validate data quality.

        Args:
            df (pd.DataFrame): Data to validate
            timeframe (str): Timeframe for validation

        Returns:
            dict: Validation results with keys: is_valid, issues
        """
        issues = []

        # Check for required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {', '.join(missing_cols)}")

        # Check for null values
        if df[required_cols].isnull().any().any():
            null_counts = df[required_cols].isnull().sum()
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        # Check OHLC logic (high >= low, etc.)
        if not (df['high'] >= df['low']).all():
            issues.append("Invalid OHLC: some highs are lower than lows")

        if not (df['high'] >= df['open']).all():
            issues.append("Invalid OHLC: some highs are lower than opens")

        if not (df['high'] >= df['close']).all():
            issues.append("Invalid OHLC: some highs are lower than closes")

        if not (df['low'] <= df['open']).all():
            issues.append("Invalid OHLC: some lows are higher than opens")

        if not (df['low'] <= df['close']).all():
            issues.append("Invalid OHLC: some lows are higher than closes")

        # Check for duplicate timestamps
        if df['time'].duplicated().any():
            dup_count = df['time'].duplicated().sum()
            issues.append(f"Duplicate timestamps: {dup_count} duplicates found")

        # Check time ordering
        if not df['time'].is_monotonic_increasing:
            issues.append("Time is not monotonically increasing")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }

    def get_summary(self):
        """
        Get summary of all cached data.

        Returns:
            pd.DataFrame: Summary table of cached data
        """
        summaries = []

        for timeframe in self.supported_timeframes:
            info = self.get_cache_info(timeframe)
            summaries.append({
                'Timeframe': timeframe,
                'Cached': '✓' if info['exists'] else '✗',
                'Candles': info['num_candles'] if info['exists'] else 0,
                'Start': info['start_time'] if info['exists'] else '-',
                'End': info['end_time'] if info['exists'] else '-'
            })

        return pd.DataFrame(summaries)


# Backward-compatible alias
TSLADataLoader = DataLoader


# Example usage
if __name__ == "__main__":
    # Initialize loader (default: TSLA)
    loader = DataLoader()

    # Example 1: Load data for a different symbol
    # aapl_loader = DataLoader(symbol='AAPL')
    # df = aapl_loader.get_data('15min')

    # Example 2: Get single timeframe (uses cache if available)
    # df = loader.get_data('15min')
    # print(df.head())

    # Example 3: Download multiple timeframes
    # timeframes = ['5min', '15min', '1h', '1day']
    # data = loader.download_all_timeframes(timeframes)

    # Example 4: Force refresh from API
    # df = loader.get_data('15min', force_refresh=True)

    # Example 5: Update existing cache
    # df = loader.update_cache('15min')

    # Example 6: Get cache summary
    summary = loader.get_summary()
    print("\n" + "="*80)
    print(f"{loader.symbol} DATA CACHE SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80 + "\n")
