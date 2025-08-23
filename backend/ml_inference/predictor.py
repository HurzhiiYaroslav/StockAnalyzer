import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
import joblib
import json
from datetime import datetime, timedelta
from src import config
from src.services import feature_engineering
from src.services import artifact_manager
from src.services.ml_utils import CoralLoss, coral_loss , coral_logits_to_probs, coral_probs_to_rank

_CACHED_ARTIFACTS = {
    "model_run_id": None, "model": None, "scaler": None,
    "params": None, "feature_columns": None
}

def _load_artifacts_if_needed(model_run_id: str):
    """Loads model and components if they are not cached or if the model_run_id has changed."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS["model_run_id"] == model_run_id and _CACHED_ARTIFACTS["model"] is not None:
        print(f"Using cached artifacts for model run: {model_run_id}")
        return

    print(f"Loading artifacts for model run: {model_run_id}...")
    run_dir = artifact_manager.get_model_run_dir(model_run_id)
    
    params, feature_columns = artifact_manager.load_parameters(model_run_id)
    
    model_path = os.path.join(run_dir, config.MODEL_FILE_NAME)
    with keras.utils.custom_object_scope({'CoralLoss': CoralLoss}):
        model = tf.keras.models.load_model(model_path)
    
    scaler_path = os.path.join(run_dir, 'standard_scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    _CACHED_ARTIFACTS = {
        "model_run_id": model_run_id, "model": model, "scaler": scaler,
        "params": params, "feature_columns": feature_columns
    }
    print("Artifacts loaded and cached successfully.")

def _prepare_features_for_prediction(raw_features_df: pd.DataFrame, model_run_id: str) -> pd.DataFrame:
    """Applies all necessary transformations to raw features before prediction."""
    _load_artifacts_if_needed(model_run_id)
    
    params = _CACHED_ARTIFACTS["params"]
    feature_columns = _CACHED_ARTIFACTS["feature_columns"]
    
    metrics_to_normalize = [col.replace('_z_score', '') for col in feature_columns if col.endswith('_z_score')]
    
    transformed_features = artifact_manager.apply_transformations(
        raw_features_df, params, metrics_to_normalize
    )

    aligned_df = pd.DataFrame(0, index=transformed_features.index, columns=feature_columns)
    common_cols = aligned_df.columns.intersection(transformed_features.columns)
    aligned_df[common_cols] = transformed_features[common_cols]
    
    meta_cols = ['date', 'ticker', 'sector']
    features_only_df = aligned_df.drop(columns=meta_cols, errors='ignore')
    
    return features_only_df

def predict(ticker_symbol: str, model_run_id: str) -> dict:
    """Generates a full prediction and display data package for a single ticker."""
    try:
        _load_artifacts_if_needed(model_run_id)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=800)
        
        raw_features_df, _ = feature_engineering.generate_features_for_ticker(
            ticker_symbol=ticker_symbol,
            start_date_str=start_date.strftime('%Y-%m-%d'),
            end_date_str=end_date.strftime('%Y-%m-%d')
        )
        if raw_features_df.empty:
            return {"error": f"Could not generate features for {ticker_symbol}."}
        
        latest_raw_features = raw_features_df.iloc[-1:]

        prepared_features = _prepare_features_for_prediction(latest_raw_features, model_run_id)

        scaler = _CACHED_ARTIFACTS["scaler"]
        scaled_features = scaler.transform(prepared_features)

        model = _CACHED_ARTIFACTS["model"]
        logits = model.predict(scaled_features)
        probs = coral_logits_to_probs(logits)
        predicted_rank = coral_probs_to_rank(probs).numpy()[0]
        
        display_cols = [
            'Return_252D', 'Price_vs_SMA200D', 'Price_vs_SMA50D', 'P_E_Ratio', 
            'ttm_eps', 'Sharpe_Ratio_252D', 'volatility_252D', 'roa_ttm', 
            'ATR_252D', 'gross_profit_margin_quarterly', 'net_profit_margin_quarterly'
        ]
        existing_display_cols = [col for col in display_cols if col in latest_raw_features.columns]
        display_features = latest_raw_features[existing_display_cols].replace({np.nan: None}).to_dict(orient='records')[0]

        return {
            "ticker": ticker_symbol,
            "model_version": model_run_id,
            "predicted_rating": int(predicted_rank),
            "prediction_probabilities": [float(p) for p in probs[0]],
            "timestamp": datetime.now().isoformat(),
            "features_for_display": display_features
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

def get_latest_model_run_id() -> str:
    """Finds the ID of the most recently trained model run."""
    try:
        all_runs = sorted([
            d for d in os.listdir(config.MODELS_DIR) 
            if os.path.isdir(os.path.join(config.MODELS_DIR, d))
        ])
        return all_runs[-1] if all_runs else None
    except (FileNotFoundError, IndexError):
        return None