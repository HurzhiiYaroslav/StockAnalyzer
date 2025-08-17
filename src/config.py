# src/config.py

import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# API CONFIGURATION
# ==============================================================================
# For https://logo.dev/
LOGO_API_KEY = os.getenv("LOGO_API_KEY")
# For https://datajockey.io/
DATAJOCKEY_API_KEY = os.getenv("DATAJOCKEY_API_KEY")
DATAJOCKEY_BASE_URL = "https://api.datajockey.io/v0/company/financials"
DATAJOCKEY_DEFAULT_PERIOD = "Q"

# ==============================================================================
# DIRECTORY AND FILE PATHS
# ==============================================================================
# Base directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINING_DIR = os.path.join(ROOT_DIR, 'training')
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')

# Subdirectories
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PRICE_CACHE_DIR = os.path.join(DATA_DIR, 'price_cache')
MODEL_SAVE_DIR = os.path.join(BACKEND_DIR, 'ml_inference/model/')
PLOTS_DIR = os.path.join(TRAINING_DIR, 'plots')

# File names
FINAL_TRAINING_DATA_FILE_NAME = 'training_data.csv'
MODEL_FILE_NAME = 'trained_nn_coral_model.keras'
COLUMNS_FILE_NAME = 'training_columns_nn.pkl'
SCALER_FILE_NAME = 'scaler_nn.pkl'

# Full file paths
FINAL_TRAINING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, FINAL_TRAINING_DATA_FILE_NAME)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILE_NAME)
COLUMNS_PATH = os.path.join(MODEL_SAVE_DIR, COLUMNS_FILE_NAME)
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, SCALER_FILE_NAME)

# ==============================================================================
# TICKER & DATE CONFIGURATION
# ==============================================================================
# Default list of tickers, used as a fallback if scraping fails
DEF_TICKERS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN', 'AVGO', 'BAC', 'BMY', 
    'BRK-B', 'CAT', 'COST', 'CRM', 'CSCO', 'CVX', 'DHR', 'DIS', 'GOOGL', 
    'GS', 'HD', 'HON', 'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'MA', 
    'MCD', 'MDT', 'MRK', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 
    'PEP', 'PFE', 'PG', 'PM', 'SBUX', 'TMO', 'TSLA', 'TXN', 'UNH', 
    'UPS', 'V', 'WMT', 'XOM'
]

def get_sp500_tickers(cache_file='sp500_tickers.csv', max_age_days=3):
    """
    Retrieves the list of S&P 500 tickers, using a local cache file to avoid
    downloading it every time.
    """
    cache_path = os.path.join(DATA_DIR, cache_file)
    
    if os.path.exists(cache_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - file_mod_time) < timedelta(days=max_age_days):
            print("Loading S&P 500 tickers from local cache file...")
            df = pd.read_csv(cache_path)
            return df['Symbol'].tolist()

    print("Downloading fresh list of S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        # Save the new list to cache file
        os.makedirs(DATA_DIR, exist_ok=True)
        sp500_table[['Symbol']].to_csv(cache_path, index=False)
        
        print(f"Successfully retrieved and cached {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        print(f"Could not retrieve S&P 500 tickers: {e}. Using default list.")
        return DEF_TICKERS
    
# Get the list of tickers dynamically
TICKERS = get_sp500_tickers()
START_DATE = "2007-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# ==============================================================================
# DATA PROCESSING AND MODEL PARAMETERS
# ==============================================================================
FUTURE_RETURN_HORIZON_DAYS = 252
NUM_CLASSES = 10