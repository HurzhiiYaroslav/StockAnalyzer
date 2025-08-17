import pandas as pd
import numpy as np

def prepare_dataframe(df_ohlcv: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
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
    for length in [10, 20, 50, 60, 100, 200, 252, 500, 750]:
        df[f'SMA_{length}'] = df['close'].rolling(window=length, min_periods=int(length*0.8)).mean()
    return df

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for length in [14, 30, 50, 100, 200, 252, 500]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        df[f'RSI_{length}'] = 100 - (100 / (1 + rs))

    for n_times in [5, 7, 9]:
        fast_period = n_times * 12
        slow_period = n_times * 26
        signal_period = n_times * 9
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        df[f'MACD_{fast_period}_{slow_period}_{signal_period}'] = ema_fast - ema_slow
        df[f'MACDs_{fast_period}_{slow_period}_{signal_period}'] = df[f'MACD_{fast_period}_{slow_period}_{signal_period}'].ewm(span=signal_period, adjust=False).mean()
        df[f'MACDh_{fast_period}_{slow_period}_{signal_period}'] = df[f'MACD_{fast_period}_{slow_period}_{signal_period}'] - df[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
    return df

def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    for n_days in [14, 20, 50, 100, 200, 252, 500]:
        df[f'ATR_{n_days}'] = true_range.rolling(n_days).mean()
    
    for length in [20, 50, 100, 200]:
        sma = df['close'].rolling(window=length).mean()
        std = df['close'].rolling(window=length).std()
        df[f'BB_upper_{length}'] = sma + (std * 2)
        df[f'BB_lower_{length}'] = sma - (std * 2)
        df[f'BB_width_{length}'] = (df[f'BB_upper_{length}'] - df[f'BB_lower_{length}']) / sma
        df[f'BB_percent_{length}'] = (df['close'] - df[f'BB_lower_{length}']) / (df[f'BB_upper_{length}'] - df[f'BB_lower_{length}'])
    return df

def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if 'volume' not in df.columns or df['volume'].isnull().all():
        return df
        
    for period in [252, 500]:
        df[f'SMA_Volume_{period}'] = df['volume'].rolling(window=period, min_periods=int(period*0.8)).mean()
        df[f'Volume_vs_SMA_Volume_{period}'] = df['volume'] / df[f'SMA_Volume_{period}']
    
    for n_days in [60, 100, 200, 252, 500]:
        signed_volume = np.sign(df['close'].diff()) * df['volume']
        df[f'OBV_{n_days}'] = signed_volume.rolling(window=n_days, min_periods=int(n_days*0.8)).sum()
    return df

def calculate_price_and_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if 'SMA_50' in df.columns: df['Price_vs_SMA50'] = df['close'] / df['SMA_50']
    if 'SMA_200' in df.columns: df['Price_vs_SMA200'] = df['close'] / df['SMA_200']
    if 'SMA_500' in df.columns: df['Price_vs_SMA500'] = df['close'] / df['SMA_500']
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns: df['SMA50_vs_SMA200'] = df['SMA_50'] / df['SMA_200']
    
    for n_days in [21, 63, 126, 252, 500, 750]:
        df[f'volatility_{n_days}d'] = df['daily_return'].rolling(window=n_days, min_periods=int(n_days*0.8)).std() * np.sqrt(252)
        df[f'Return_{n_days}D'] = df['close'].pct_change(periods=n_days)

    for window in [60, 120, 252, 500]:
        rolling_mean_return = df['daily_return'].rolling(window=window).mean()
        rolling_std_return = df['daily_return'].rolling(window=window).std()
        df[f'Sharpe_Ratio_{window}D'] = (rolling_mean_return / rolling_std_return) * np.sqrt(252)
    return df

def calculate_new_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for length in [14, 50, 100, 200]:
        low_min = df['low'].rolling(window=length).min()
        high_max = df['high'].rolling(window=length).max()
        df[f'Stoch_K_{length}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'Stoch_D_{length}'] = df[f'Stoch_K_{length}'].rolling(window=3).mean()

    for length in [14, 50, 100]:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[(plus_dm < 0) | (plus_dm < -minus_dm)] = 0
        minus_dm[(minus_dm < 0) | (minus_dm > -plus_dm)] = 0
        
        tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df[f'ADX_{length}'] = dx.ewm(alpha=1/length, adjust=False).mean()
    return df

def clean_and_fill_features(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'dividends', 'stock splits', 'ticker', 'daily_return']
    feature_cols = [col for col in df.columns if col not in original_cols]
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in feature_cols:
        if df[col].isna().any():
            if any(x in col.lower() for x in ['sma', 'rsi', 'atr', 'bb', 'macd', 'stoch', 'adx']):
                df[col] = df[col].ffill()
            else:
                df[col] = df[col].bfill().ffill()
    
    df = df.dropna()
    return df

def calculate_historical_features_for_ticker(df_ohlcv: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    print(f"  Calculating technical features for {ticker_symbol}...", end=" ", flush=True)
    if df_ohlcv.empty or len(df_ohlcv) < 750:
        print(f"Not enough data ({len(df_ohlcv)} rows, need at least 750 for long-term indicators).")
        return pd.DataFrame()
    
    df = prepare_dataframe(df_ohlcv, ticker_symbol)
    if df.empty: return pd.DataFrame()
    
    df = calculate_trend_indicators(df)
    df = calculate_momentum_indicators(df)
    df = calculate_volatility_indicators(df)
    df = calculate_volume_indicators(df)
    df = calculate_price_and_risk_indicators(df)
    df = calculate_new_advanced_indicators(df)
    
    df = clean_and_fill_features(df)
    
    if df.empty:
        print("    No data after calculation/cleaning.")
    else:
        print(f"    OK ({len(df)} rows)")
    return df