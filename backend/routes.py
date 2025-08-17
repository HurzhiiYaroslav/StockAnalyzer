from flask import Blueprint, jsonify, request
from src.services.process_data import get_features_for_ticker
from src.services.fetch_data import get_company_info, get_price_history_for_chart
from ml_inference.predictor import predict_from_features
import numpy as np
import yfinance as yf
import os, json

api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Missing ticker parameter'}), 400
            
        ticker = data['ticker']
        print(f"Received prediction request for ticker: {ticker}")
        
        features_df = get_features_for_ticker(ticker)
        if features_df.empty:
            return jsonify({'error': f'Could not get features for {ticker}'}), 404
            
        company_info = get_company_info(ticker)
        if not company_info:
            return jsonify({'error': f'Could not get company info for {ticker}'}), 404

        ticker_info = yf.Ticker(ticker).info
        
        if 'P_E_Ratio' not in features_df.columns or features_df['P_E_Ratio'].isna().any():
            features_df['P_E_Ratio'] = ticker_info.get('trailingPE')
        if 'ttm_eps' not in features_df.columns or features_df['ttm_eps'].isna().any():
            features_df['ttm_eps'] = ticker_info.get('trailingEps')
            
        price_history = get_price_history_for_chart(ticker)
        
        prediction = predict_from_features(features_df)
        if prediction is None:
            return jsonify({'error': f'Could not make prediction for {ticker}'}), 500

        selected_columns = [
            'Return_250D', 'Price_vs_SMA200', 'Price_vs_SMA50', 
            'P_E_Ratio', 'ttm_eps', 'Sharpe_Ratio_252D', 'volatility_250d',
            'roa_ttm', 'ATR_252D','gross_profit_margin_quarterly','net_profit_margin_quarterly'
        ]
        features_df = features_df[selected_columns]
        features_dict = features_df.replace({np.nan: None}).to_dict(orient='records')
            
        return jsonify({
            'ticker': ticker,
            'prediction': prediction,
            'company_info': company_info,
            'price_history': price_history,
            'features': features_dict
        })
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api.route('/tickers/search', methods=['GET'])
def search_tickers():
    try:
        query = request.args.get('query', '').strip()
        if not query:
            return jsonify({"tickers": []})

        raw_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
        ticker_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
        
        all_local_tickers = []
        for fname in ticker_files:
            with open(os.path.join(raw_dir, fname), encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    info = data.get('company_info', {})
                    symbol = info.get('ticker', fname.replace('.json', '')).upper()
                    name = info.get('name', symbol)
                    all_local_tickers.append({"symbol": symbol, "name": name})
                except Exception:
                    continue
        
        results = []
        added_symbols = set()

        for ticker in all_local_tickers:
            if query.lower() in ticker['symbol'].lower() or query.lower() in ticker['name'].lower():
                if ticker['symbol'] not in added_symbols:
                    results.append(ticker)
                    added_symbols.add(ticker['symbol'])
        
        if len(results) < 5:
            try:
                if query.upper() not in added_symbols:
                    ticker_yf = yf.Ticker(query.upper())
                    info_yf = ticker_yf.info
                    if info_yf and info_yf.get('quoteType') != "MUTUALFUND" and info_yf.get('regularMarketPrice') is not None:
                        symbol_yf = query.upper()
                        name_yf = info_yf.get('longName', info_yf.get('shortName', symbol_yf))
                        
                        if symbol_yf not in added_symbols:
                            results.append({
                                "symbol": symbol_yf,
                                "name": name_yf
                            })
                            added_symbols.add(symbol_yf)
            except Exception:
                pass
        
        final_results = []
        final_added_symbols = set()
        for res_ticker in results:
            if res_ticker['symbol'].upper() not in final_added_symbols:
                final_results.append(res_ticker)
                final_added_symbols.add(res_ticker['symbol'].upper())

        return jsonify({"tickers": final_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/tickers', methods=['GET'])
def get_tickers():
    try:
        tickers = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "V", "name": "Visa Inc."},
            {"symbol": "WMT", "name": "Walmart Inc."},
            {"symbol": "UNH", "name": "UnitedHealth Group Incorporated"},
            {"symbol": "PG", "name": "Procter & Gamble Co."},
            {"symbol": "JNJ", "name": "Johnson & Johnson"},
            {"symbol": "MA", "name": "Mastercard Incorporated"},
            {"symbol": "HD", "name": "Home Depot Inc."}
        ]
        return jsonify({"tickers": tickers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
