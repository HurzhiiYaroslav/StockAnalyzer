import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

try:
    from src import config
except ImportError:
    print("ERROR: Could not import config.py. Make sure this script is run from the project's root directory.")
    exit()

def analyze_feature_importance(model, X_test, y_test_scalar, feature_names, model_type="keras_coral", save_path=None):
    print("  Calculating baseline performance for feature importance...")

    def predict_fn(X):
        if model_type == "keras_coral":
            from src.services.ml_utils import coral_logits_to_probs, coral_probs_to_rank
            logits = model.predict(X, verbose=0)
            probs = coral_logits_to_probs(logits)
            return coral_probs_to_rank(probs).numpy()
        else:
            return model.predict(X)

    baseline_predictions = predict_fn(X_test)
    baseline_mae = mean_absolute_error(y_test_scalar, baseline_predictions)
    print(f"    Baseline MAE: {baseline_mae:.4f}")

    importances = []
    print("  Calculating permutation importance for each feature...")
    for i, col in enumerate(feature_names):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        permuted_predictions = predict_fn(X_test_permuted)
        permuted_mae = mean_absolute_error(y_test_scalar, permuted_predictions)
        importance = permuted_mae - baseline_mae
        importances.append(importance)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\n  Top 20 most important features:")
    print(importance_df.head(20))

    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=importance_df.head(30), palette='viridis')
    plt.title('Top 30 Feature Importances (Permutation Method)')
    plt.xlabel('Importance (Increase in MAE)')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if not save_path:
        print("Warning: Save path not provided. Plot will not be saved.")
        plt.show()
        return
        
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Feature importance plot saved to: {save_path}")
    except Exception as e:
        print(f"  Error saving feature importance plot: {e}")
    plt.close()

def find_latest_model_dir(models_base_dir):
    if not os.path.isdir(models_base_dir):
        return None
    
    all_versions = [d for d in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, d))]
    
    if not all_versions:
        return None
        
    latest_version_dir = sorted(all_versions)[-1]
    return os.path.join(models_base_dir, latest_version_dir)

if __name__ == "__main__":
    print("--- Starting Manual Feature Importance Analysis ---")

    try:
        print(f"Searching for models in directory: {config.MODELS_DIR}")
        latest_model_dir = find_latest_model_dir(config.MODELS_DIR)
        
        if not latest_model_dir:
            raise FileNotFoundError(f"No model versions found in {config.MODELS_DIR}")
            
        print(f"Found latest model version: {os.path.basename(latest_model_dir)}")
        
        model_path = os.path.join(latest_model_dir, config.MODEL_FILE_NAME)
        features_path = os.path.join(latest_model_dir, config.FEATURE_COLUMNS_FILE_NAME)
        
        x_test_path = os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy')
        y_test_path = os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy')
        
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")

        print("Loading test data...")
        X_test = np.load(x_test_path)
        y_test_scalar = np.load(y_test_path)
        print("Test data loaded successfully.")

        print("Loading feature names...")
        with open(features_path, 'r') as f:
            feature_names = json.load(f)['columns']
        print(f"Loaded {len(feature_names)} feature names.")

        print("\n--- Running Analysis ---")
        analyze_feature_importance(
            model=model,
            X_test=X_test,
            y_test_scalar=y_test_scalar,
            feature_names=feature_names,
            model_type="keras_coral",
            save_path=os.path.join(config.PLOTS_DIR, f"feature_importance_{os.path.basename(latest_model_dir)}.png")
        )

        print("\n--- Analysis complete. ---")

    except FileNotFoundError as e:
        print(f"\nERROR: File not found. Please check your paths in 'src/config.py' and ensure artifacts exist.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")