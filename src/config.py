import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# API CONFIGURATION
# ==============================================================================
LOGO_API_KEY = os.getenv("LOGO_API_KEY")
DATAJOCKEY_API_KEY = os.getenv("DATAJOCKEY_API_KEY")
DATAJOCKEY_BASE_URL = "https://api.datajockey.io/v0/company/financials"
DATAJOCKEY_DEFAULT_PERIOD = "Q"

# ==============================================================================
# DIRECTORY AND FILE PATHS
# ==============================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINING_DIR = os.path.join(ROOT_DIR, 'training')
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PRICE_CACHE_DIR = os.path.join(DATA_DIR, 'price_cache')
MODELS_DIR = os.path.join(BACKEND_DIR, 'ml_inference/models/')
PLOTS_DIR = os.path.join(TRAINING_DIR, 'plots')

# --- File Names ---
FINAL_TRAINING_DATA_FILE_NAME = 'training_data.csv'
MODEL_FILE_NAME = 'trained_nn_coral_model.keras'
NORMALIZATION_PARAMS_FILE_NAME = 'normalization_parameters.feather'
FEATURE_COLUMNS_FILE_NAME = 'feature_columns.json'

FINAL_TRAINING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, FINAL_TRAINING_DATA_FILE_NAME)

# ==============================================================================
# TICKER & DATE CONFIGURATION
# ==============================================================================
DEF_TICKERS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN', 'AVGO', 'BAC', 'BMY', 
    'BRK-B', 'CAT', 'COST', 'CRM', 'CSCO', 'CVX', 'DHR', 'DIS', 'GOOGL', 
    'GS', 'HD', 'HON', 'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'MA', 
    'MCD', 'MDT', 'MRK', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 
    'PEP', 'PFE', 'PG', 'PM', 'SBUX', 'TMO', 'TSLA', 'TXN', 'UNH', 
    'UPS', 'V', 'WMT', 'XOM'
]

def get_sp500_tickers(cache_file='sp500_tickers.csv', max_age_days=3):
    cache_path = os.path.join(DATA_DIR, cache_file)
    
    if os.path.exists(cache_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - file_mod_time) < timedelta(days=max_age_days):
            df = pd.read_csv(cache_path)
            return df['Symbol'].tolist()
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        os.makedirs(DATA_DIR, exist_ok=True)
        sp500_table[['Symbol']].to_csv(cache_path, index=False)
        
        print(f"Successfully retrieved and cached {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        print(f"Could not retrieve S&P 500 tickers: {e}. Using default list.")
        return DEF_TICKERS
    
TICKERS = get_sp500_tickers()
START_DATE = "2007-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# ==============================================================================
# TECHNICAL INDICATOR PARAMETERS
# ==============================================================================
TREND_PERIODS = [10, 20, 50, 60, 100, 200, 252, 504]
MOMENTUM_PERIODS = [14, 30, 50, 100, 200, 252, 504]
VOLATILITY_PERIODS = [14, 20, 50, 100, 200, 252, 504]
BOLLINGER_BANDS_PERIODS = [20, 50, 100, 200]
VOLUME_PERIODS = [252, 504]
OBV_PERIODS = [60, 100, 200, 252, 504]
PRICE_RISK_PERIODS = [21, 63, 126, 252, 504]
SHARPE_RATIO_PERIODS = [60, 120, 252, 504]
STOCHASTIC_PERIODS = [14, 50, 100, 200]
ADX_PERIODS = [14, 50, 100]
MACD_MULTIPLIERS = [1, 3, 5, 7, 9]

# ==============================================================================
# DATA PROCESSING AND MODEL PARAMETERS
# ==============================================================================
FUTURE_RETURN_HORIZON_DAYS = 252
NUM_CLASSES = 5
TECHNICAL_INDICATOR_PREFIXES  = ('SMA', 'RSI', 'ATR', 'BB', 'MACD', 'Stoch', 'ADX', 'Price_vs', 'Return', 'volatility', 'Sharpe', 'Volume', 'OBV')