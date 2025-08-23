import pandas as pd
import numpy as np
import os
import json
from src import config

def get_model_run_dir(model_run_id: str) -> str:
    """Returns the full path to the artifact directory for a specific model run."""
    return os.path.join(config.MODELS_DIR, model_run_id)

def create_and_save_parameters(df_train: pd.DataFrame, metrics_to_normalize: list, feature_columns: list, model_run_id: str):
    """Calculates and saves normalization parameters based *only* on the training data."""
    print(f"--- Creating and saving artifacts for model run: {model_run_id} ---")
    
    run_dir = get_model_run_dir(model_run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    params_file_path = os.path.join(run_dir, config.NORMALIZATION_PARAMS_FILE_NAME)
    feature_columns_path = os.path.join(run_dir, config.FEATURE_COLUMNS_FILE_NAME)

    if 'date' not in df_train.index.names and 'date' in df_train.columns:
        df_train = df_train.set_index('date')

    zscore_params_list = []
    for metric in metrics_to_normalize:
        if metric in df_train.columns:
            grouped = df_train.groupby('sector')[metric]
            sector_mean = grouped.mean().rename(f"{metric}_mean")
            sector_std = grouped.std().rename(f"{metric}_std")
            zscore_params_list.append(pd.DataFrame({f"{metric}_mean": sector_mean, f"{metric}_std": sector_std}))

    all_params = pd.concat(zscore_params_list, axis=1)
    
    all_params.reset_index().to_feather(params_file_path)
    print(f"Normalization parameters saved to {params_file_path}")

    with open(feature_columns_path, 'w') as f:
        json.dump({"columns": feature_columns}, f, indent=4)
    print(f"Feature columns saved to {feature_columns_path}")

def load_parameters(model_run_id: str) -> tuple:
    """Loads normalization parameters and feature columns from a specific model run directory."""
    print(f"--- Loading artifacts for model run: {model_run_id} ---")
    
    run_dir = get_model_run_dir(model_run_id)
    params_file_path = os.path.join(run_dir, config.NORMALIZATION_PARAMS_FILE_NAME)
    feature_columns_path = os.path.join(run_dir, config.FEATURE_COLUMNS_FILE_NAME)

    if not os.path.exists(params_file_path) or not os.path.exists(feature_columns_path):
        raise FileNotFoundError(f"Could not find model artifacts in directory: {run_dir}.")
    
    params_df = pd.read_feather(params_file_path).set_index('sector')
    
    with open(feature_columns_path, 'r') as f:
        feature_columns_data = json.load(f)
        feature_columns = feature_columns_data.get("columns", feature_columns_data)
        
    print("Normalization parameters and feature columns loaded successfully.")
    return params_df, feature_columns

def apply_transformations(df: pd.DataFrame, params: pd.DataFrame, metrics_to_normalize: list) -> pd.DataFrame:
    """Applies pre-calculated Z-score transformations to a new DataFrame using vectorized operations."""
    df_transformed = df.copy()
    
    df_transformed = df_transformed.merge(params, on='sector', how='left')
    
    for metric in metrics_to_normalize:
        if metric in df_transformed.columns:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            z_score_col = f"{metric}_z_score"
            epsilon = 1e-6
            
            df_transformed[z_score_col] = (df_transformed[metric] - df_transformed[mean_col]) / (df_transformed[std_col] + epsilon)
            df_transformed[z_score_col] = df_transformed[z_score_col].fillna(0)

    param_cols_to_drop = [col for col in df_transformed.columns if '_mean' in col or '_std' in col]
    df_transformed.drop(columns=param_cols_to_drop, inplace=True)

    return df_transformed