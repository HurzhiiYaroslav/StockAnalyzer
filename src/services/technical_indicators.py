import pandas as pd
import numpy as np
from src import config

def prepare_dataframe(df_ohlcv: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    """Prepares the initial OHLCV DataFrame by adding a ticker symbol and calculating daily returns."""
    try:
        df = df_ohlcv.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"  Missing one of OHLCV columns for {ticker_symbol}. Available: {df.columns.tolist()}")
            return pd.DataFrame()
        
        df['ticker'] = ticker_symbol
        df['daily_return'] = df['close'].pct_change()
        return df
        
    except Exception as e:
        print(f"  Error preparing DataFrame for {ticker_symbol}: {e}")
        return pd.DataFrame()

def calculate_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates SMAs and Bollinger Bands, as they both use SMA and standard deviation."""
    for length in config.TREND_PERIODS:
        df[f'SMA_{length}D'] = df['close'].rolling(window=length, min_periods=int(length*0.8)).mean()
    
    for length in config.BOLLINGER_BANDS_PERIODS:
        sma = df.get(f'SMA_{length}D')
        if sma is None:
            sma = df['close'].rolling(window=length).mean()
        
        std = df['close'].rolling(window=length).std()
        df[f'BB_upper_{length}D'] = sma + (std * 2)
        df[f'BB_lower_{length}D'] = sma - (std * 2)
        df[f'BB_width_{length}D'] = (df[f'BB_upper_{length}D'] - df[f'BB_lower_{length}D']) / sma
        df[f'BB_percent_{length}D'] = (df['close'] - df[f'BB_lower_{length}D']) / (df[f'BB_upper_{length}D'] - df[f'BB_lower_{length}D'])
    return df

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates momentum indicators: RSI, MACD, Stochastic Oscillator, and ADX."""
    for length in config.MOMENTUM_PERIODS:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        df[f'RSI_{length}D'] = 100 - (100 / (1 + rs))

    for n_times in config.MACD_MULTIPLIERS:
        fast_period = n_times * 12
        slow_period = n_times * 26
        signal_period = n_times * 9
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        df[f'MACD_{fast_period}_{slow_period}_{signal_period}'] = ema_fast - ema_slow
        df[f'MACDs_{fast_period}_{slow_period}_{signal_period}'] = df[f'MACD_{fast_period}_{slow_period}_{signal_period}'].ewm(span=signal_period, adjust=False).mean()
        df[f'MACDh_{fast_period}_{slow_period}_{signal_period}'] = df[f'MACD_{fast_period}_{slow_period}_{signal_period}'] - df[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
    
    for length in config.STOCHASTIC_PERIODS:
        low_min = df['low'].rolling(window=length).min()
        high_max = df['high'].rolling(window=length).max()
        df[f'Stoch_K_{length}D'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'Stoch_D_{length}D'] = df[f'Stoch_K_{length}D'].rolling(window=3).mean()

    for length in config.ADX_PERIODS:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[(plus_dm < 0) | (plus_dm < -minus_dm)] = 0
        minus_dm[(minus_dm < 0) | (minus_dm > -plus_dm)] = 0
        
        tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
        
        dx_denominator = (plus_di + minus_di)
        dx = 100 * (abs(plus_di - minus_di) / dx_denominator.replace(0, np.nan))
        
        df[f'ADX_{length}D'] = dx.ewm(alpha=1/length, adjust=False).mean()
    return df

def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Average True Range (ATR). Bollinger Bands are now in calculate_trend_indicators."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    for n_days in config.VOLATILITY_PERIODS:
        df[f'ATR_{n_days}D'] = true_range.rolling(n_days).mean()
    return df

def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates volume-based indicators: SMA of volume and On-Balance Volume (OBV)."""
    if 'volume' not in df.columns or df['volume'].isnull().all():
        return df
        
    for period in config.VOLUME_PERIODS:
        df[f'SMA_Volume_{period}D'] = df['volume'].rolling(window=period, min_periods=int(period*0.8)).mean()
        df[f'Volume_vs_SMA_Volume_{period}D'] = df['volume'] / df[f'SMA_Volume_{period}D']
    
    for n_days in config.OBV_PERIODS:
        signed_volume = np.sign(df['close'].diff()) * df['volume']
        df[f'OBV_{n_days}D'] = signed_volume.rolling(window=n_days, min_periods=int(n_days*0.8)).sum()
    return df

def calculate_price_and_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates price-relative and risk-adjusted indicators like Price vs. SMA and Sharpe Ratio."""
    if 'SMA_50D' in df.columns: df['Price_vs_SMA50D'] = df['close'] / df['SMA_50D']
    if 'SMA_200D' in df.columns: df['Price_vs_SMA200D'] = df['close'] / df['SMA_200D']
    if 'SMA_500D' in df.columns: df['Price_vs_SMA500D'] = df['close'] / df['SMA_500D']
    if 'SMA_50D' in df.columns and 'SMA_200D' in df.columns: df['SMA50D_vs_SMA200D'] = df['SMA_50D'] / df['SMA_200D']
    
    for n_days in config.PRICE_RISK_PERIODS:
        df[f'volatility_{n_days}D'] = df['daily_return'].rolling(window=n_days, min_periods=int(n_days*0.8)).std() * np.sqrt(252)
        df[f'Return_{n_days}D'] = df['close'].pct_change(periods=n_days)

    for window in config.SHARPE_RATIO_PERIODS:
        rolling_mean_return = df['daily_return'].rolling(window=window).mean()
        rolling_std_return = df['daily_return'].rolling(window=window).std()
        df[f'Sharpe_Ratio_{window}D'] = (rolling_mean_return / rolling_std_return) * np.sqrt(252)
    return df

def clean_and_fill_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the DataFrame by handling infinite values and filling NaNs using a forward-fill only strategy."""
    original_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'dividends', 'stock splits', 'ticker', 'daily_return']
    feature_cols = [col for col in df.columns if col not in original_cols]
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df[feature_cols] = df[feature_cols].ffill()
    
    df = df.dropna()
    return df

def calculate_historical_features_for_ticker(df_ohlcv: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    """Orchestrates the entire feature calculation pipeline for a single ticker."""
    print(f"  Calculating technical features for {ticker_symbol}...", end=" ", flush=True)
    
    df = prepare_dataframe(df_ohlcv, ticker_symbol)
    if df.empty: return pd.DataFrame()
    
    df = calculate_trend_indicators(df)
    df = calculate_momentum_indicators(df)
    df = calculate_volatility_indicators(df)
    df = calculate_volume_indicators(df)
    df = calculate_price_and_risk_indicators(df)
    
    df = clean_and_fill_features(df)
    
    if df.empty:
        print("    No data after calculation/cleaning.")
    else:
        print(f"    OK ({len(df)} rows)")
    return df