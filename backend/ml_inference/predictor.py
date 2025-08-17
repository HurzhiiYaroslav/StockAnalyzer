import pandas as pd
import tensorflow as tf
import joblib
from src import config

_model = None
_scaler = None
_training_columns = None

@tf.keras.utils.register_keras_serializable()
def coral_loss(y_true, y_pred): return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True))

def coral_logits_to_probs(logits):
    return tf.math.sigmoid(logits)

def coral_probs_to_rank(probs):
    return tf.reduce_sum(tf.cast(probs > 0.5, dtype=tf.int32), axis=1) + 1

def _load_model_and_components():
    global _model, _scaler, _training_columns
    if _model is None:
        print("Loading model and components for the first time...")
        _model = tf.keras.models.load_model(config.MODEL_PATH, custom_objects={'coral_loss': coral_loss})
        _scaler = joblib.load(config.SCALER_PATH)
        _training_columns = joblib.load(config.COLUMNS_PATH)
    return _model, _scaler, _training_columns

@tf.function(reduce_retracing=True)
def _predict_step(X_scaled):
    model, _, _ = _load_model_and_components()
    logits = model(X_scaled, training=False)
    probs = coral_logits_to_probs(logits)
    return coral_probs_to_rank(probs)

def predict_from_features(features_df: pd.DataFrame) -> int:
    try:
        _, scaler, training_columns = _load_model_and_components()

        X = pd.DataFrame(columns=training_columns, index=features_df.index)

        common_columns = X.columns.intersection(features_df.columns)
        X[common_columns] = features_df[common_columns]

        missing_columns = set(training_columns) - set(common_columns)
        if missing_columns:
            print(f"Warning: Missing columns will be imputed: {missing_columns}")

        X_scaled_partial = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled_partial, columns=training_columns, index=X.index)
        X_scaled_df.fillna(0, inplace=True)
        
        predicted_ratings = _predict_step(X_scaled_df.values)
        return int(predicted_ratings[0])
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return 0