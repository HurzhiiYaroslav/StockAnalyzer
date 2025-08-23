from flask import Blueprint, jsonify, request
from src.services.fetch_data import get_company_info, get_price_history_for_chart
from src.services import metadata_service
from .ml_inference import predictor
from src import config
import traceback

api = Blueprint('api', __name__)

LATEST_MODEL_RUN_ID = predictor.get_latest_model_run_id()

@api.route('/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint to get a full analysis package for a given ticker."""
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Missing ticker parameter'}), 400
            
        ticker = data['ticker'].upper()
        print(f"Received prediction request for ticker: {ticker}")

        if not LATEST_MODEL_RUN_ID:
             return jsonify({'error': 'No trained model available to make a prediction.'}), 503

        prediction_result = predictor.predict(ticker, LATEST_MODEL_RUN_ID)
        
        if 'error' in prediction_result:
            print(f"Error during prediction for {ticker}: {prediction_result['error']}")
            return jsonify({'error': prediction_result['error']}), 404

        company_info = get_company_info(ticker)
        price_history = get_price_history_for_chart(ticker)

        return jsonify({
            'prediction': prediction_result,
            'company_info': company_info,
            'price_history': price_history
        })
        
    except Exception as e:
        print(f"Critical error in /predict endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

@api.route('/tickers/search', methods=['GET'])
def search_tickers():
    """Endpoint for ticker autocompletion search."""
    try:
        query = request.args.get('query', '').strip().lower()
        if not query:
            return jsonify({"tickers": []})

        metadata_cache = metadata_service.load_cache()
        results = []
        for symbol, info in metadata_cache.items():
            name = info.get('name', '')
            if query in symbol.lower() or query in name.lower():
                results.append({"symbol": symbol, "name": name})

        results.sort(key=lambda x: x['name'])
        return jsonify({"tickers": results[:10]})
    except Exception as e:
        print(f"Error in /tickers/search endpoint: {str(e)}")
        return jsonify({"error": "An internal server error occurred."}), 500

@api.route('/tickers', methods=['GET'])
def get_tickers():
    """Endpoint to get a default list of popular tickers."""
    try:
        tickers_list = [{"symbol": t, "name": ""} for t in config.DEF_TICKERS]
        metadata_cache = metadata_service.load_cache()
        for item in tickers_list:
            if item['symbol'] in metadata_cache:
                item['name'] = metadata_cache[item['symbol']].get('name', '')
        return jsonify({"tickers": tickers_list})
    except Exception as e:
        print(f"Error in /tickers endpoint: {str(e)}")
        return jsonify({"error": "An internal server error occurred."}), 500