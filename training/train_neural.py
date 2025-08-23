import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import joblib
import os
import numpy as np
from datetime import datetime
import json
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from src.services import artifact_manager
from src.services.ml_utils import to_coral_format, coral_loss, coral_logits_to_probs, coral_probs_to_rank, coral_probs_to_soft_rank

def load_and_prepare_data_nn(file_path):
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None, None, None
    
    print(f"Loaded {len(data)} rows.")
    if 'Rating' not in data.columns:
        print("Error: 'Rating' column is missing.")
        return None, None, None, None
    
    X = data.drop(columns=['Rating'])
    y_scalar = data['Rating'].astype(int)

    if X.empty or y_scalar.empty:
        print("X or y arrays are empty.")
        return None, None, None, None
        
    training_columns = X.columns.tolist()
    y_coral = to_coral_format(y_scalar.values, config.NUM_CLASSES)
    
    return X, y_coral, y_scalar, training_columns

def build_model_for_tuner(hp, input_shape, num_classes):
    num_outputs = num_classes - 1
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=8, step=1)
    for i in range(num_hidden_layers):
        with hp.conditional_scope("num_hidden_layers", list(range(1, 9))):
            if i < num_hidden_layers:
                model.add(layers.Dense(
                    units=hp.Int(f'units_layer_{i}', min_value=32, max_value=1024, step=32),
                    activation=hp.Choice(f'activation_layer_{i}', values=['relu', 'elu', 'tanh', 'selu', 'gelu']),
                    kernel_initializer=hp.Choice(f'initializer_layer_{i}', values=['he_normal', 'glorot_uniform', 'lecun_normal']),
                    kernel_regularizer=keras.regularizers.l2(hp.Float(f'l2_kernel_layer_{i}', min_value=1e-5, max_value=1e-2, sampling='log'))
                ))
                if hp.Boolean(f'batch_norm_layer_{i}', default=False):
                    model.add(layers.BatchNormalization())
                model.add(layers.Dropout(rate=hp.Float(f'dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.05)))
    model.add(layers.Dense(num_outputs))

    optimizer_choice = hp.Choice('optimizer', ['adam', 'adamw', 'nadam', 'rmsprop'])
    use_lr_schedule = hp.Boolean('use_lr_schedule', default=False)
    
    if use_lr_schedule:
        initial_learning_rate = hp.Float("initial_learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
        decay_steps = 1000
        decay_rate = hp.Float("decay_rate", min_value=0.85, max_value=0.99)
        learning_rate = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    else:
        learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=hp.Float('weight_decay_adamw', min_value=1e-5, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=hp.Float('rmsprop_rho', min_value=0.8, max_value=0.99, default=0.9))
                                             
    model.compile(optimizer=optimizer, loss=coral_loss, metrics=['accuracy'])
    return model

def train_evaluate_save_nn_model(X_df, y_coral, y_scalar_for_eval, initial_columns, model_run_id):
    if X_df is None: return

    run_dir = artifact_manager.get_model_run_dir(model_run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Artifacts for this run will be saved to: {run_dir}")

    print("Splitting data...")
    X_with_meta = pd.concat([X_df, y_scalar_for_eval.to_frame(name='Rating')], axis=1)
    
    train_val_df, test_df = train_test_split(
        X_with_meta, test_size=0.2, random_state=42, stratify=X_with_meta['Rating']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['Rating']
    )
    
    y_train_scalar = train_df.pop('Rating')
    y_val_scalar = val_df.pop('Rating')
    y_test_scalar = test_df.pop('Rating')
    
    X_train_df = train_df
    X_val_df = val_df
    X_test_df = test_df

    y_train_coral = to_coral_format(y_train_scalar.values, config.NUM_CLASSES)
    y_val_coral = to_coral_format(y_val_scalar.values, config.NUM_CLASSES)
    
    technical_cols = [col for col in X_train_df.columns if col.startswith(config.TECHNICAL_INDICATOR_PREFIXES)]
    meta_cols = ['date', 'ticker', 'sector']
    fundamental_metrics = [
        col for col in X_train_df.columns 
        if col not in technical_cols and col not in meta_cols
    ]
    all_current_columns = X_train_df.columns.tolist()

    print(f"Dynamically identified {len(fundamental_metrics)} fundamental metrics for Z-score normalization.")
    print(f"Creating and saving normalization parameters based on the training set...")
    artifact_manager.create_and_save_parameters(X_train_df, fundamental_metrics,all_current_columns, model_run_id)
    
    params, _ = artifact_manager.load_parameters(model_run_id)
    
    print("Applying transformations to train, validation, and test sets...")
    X_train_transformed = artifact_manager.apply_transformations(X_train_df, params, fundamental_metrics)
    X_val_transformed = artifact_manager.apply_transformations(X_val_df, params, fundamental_metrics)
    X_test_transformed = artifact_manager.apply_transformations(X_test_df, params, fundamental_metrics)
    
    X_train_transformed.drop(columns=fundamental_metrics, inplace=True, errors='ignore')
    X_val_transformed.drop(columns=fundamental_metrics, inplace=True, errors='ignore')
    X_test_transformed.drop(columns=fundamental_metrics, inplace=True, errors='ignore')
    
    final_feature_columns = X_train_transformed.columns.tolist()
    columns_path = os.path.join(run_dir, config.FEATURE_COLUMNS_FILE_NAME)
    with open(columns_path, 'w') as f:
        json.dump({"columns": final_feature_columns}, f, indent=4)
    print(f"Final feature columns saved to: {columns_path}")

    X_train_transformed = X_train_transformed.drop(columns=meta_cols, errors='ignore')
    X_val_transformed = X_val_transformed.drop(columns=meta_cols, errors='ignore')
    X_test_transformed = X_test_transformed.drop(columns=meta_cols, errors='ignore')

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    X_val_scaled = scaler.transform(X_val_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)

    print("Searching for hyperparameters with Keras Tuner...")
    tuner_dir = os.path.join(config.TRAINING_DIR, 'keras_tuner_dir')
    tuner = kt.Hyperband(
        lambda hp: build_model_for_tuner(hp, (X_train_scaled.shape[1],), config.NUM_CLASSES),
        objective='val_loss', max_epochs=15, factor=3, directory=tuner_dir,
        project_name=f"stock_coral_{model_run_id}"
    )
    stop_early_tuner = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    BATCH_SIZE = 512
    tuner_train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_coral)).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    tuner_val_ds = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_coral)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    tuner.search(tuner_train_ds, epochs=30, validation_data=tuner_val_ds, callbacks=[stop_early_tuner], verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    final_model = build_model_for_tuner(best_hps, (X_train_scaled.shape[1],), config.NUM_CLASSES)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_coral)).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_coral)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    early_stopping_final = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = final_model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[early_stopping_final], verbose=1)

    print("\nEvaluating the final model...")
    test_logits = final_model.predict(X_test_scaled, verbose=0)
    test_probs = coral_logits_to_probs(test_logits)

    print("\n--- Hard Ranking (Threshold > 0.5) Evaluation ---")
    y_pred_hard = coral_probs_to_rank(test_probs).numpy()
    mae_hard = mean_absolute_error(y_test_scalar, y_pred_hard)
    accuracy_exact_hard = accuracy_score(y_test_scalar, y_pred_hard)
    within_one_accuracy_hard = np.mean(np.abs(y_test_scalar.values - y_pred_hard) <= 1)
    print(f"  Hard Rank MAE: {mae_hard:.4f}")
    print(f"  Hard Rank Accuracy (exact): {accuracy_exact_hard:.4f}")
    print(f"  Hard Rank Accuracy (+/- 1 score): {within_one_accuracy_hard:.4f}")

    print("\n--- Soft Ranking (Sum of Probabilities) Evaluation ---")
    y_pred_soft_rounded = coral_probs_to_soft_rank(test_probs).numpy()
    mae_soft = mean_absolute_error(y_test_scalar, y_pred_soft_rounded)
    accuracy_exact_soft = accuracy_score(y_test_scalar, y_pred_soft_rounded)
    within_one_accuracy_soft = np.mean(np.abs(y_test_scalar.values - y_pred_soft_rounded) <= 1)
    print(f"  Soft Rank MAE: {mae_soft:.4f}")
    print(f"  Soft Rank Accuracy (exact): {accuracy_exact_soft:.4f}")
    print(f"  Soft Rank Accuracy (+/- 1 score): {within_one_accuracy_soft:.4f}")
    
    print("\n--- Comparison Summary ---")
    print(f"MAE improvement with Soft Ranking: {mae_hard - mae_soft:.4f} (lower is better)")
    
    best_y_pred = y_pred_soft_rounded if mae_soft < mae_hard else y_pred_hard
    best_method_name = "Soft Ranking" if mae_soft < mae_hard else "Hard Ranking"

    print(f"\nGenerating Confusion Matrix for the best method: {best_method_name}")
    labels_nn = sorted(list(set(y_test_scalar.unique()) | set(np.unique(best_y_pred))))
    cm_nn = confusion_matrix(y_test_scalar, best_y_pred, labels=labels_nn)
    cm_df_nn = pd.DataFrame(cm_nn, index=[f"{i}" for i in labels_nn], columns=[f"{i}" for i in labels_nn])
    print(cm_df_nn)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df_nn, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Run: {model_run_id})', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"    Confusion matrix saved to: {plot_path}")

    print("\nSaving model and components...")
    try:
        model_path = os.path.join(run_dir, config.MODEL_FILE_NAME)
        scaler_path = os.path.join(run_dir, 'standard_scaler.joblib')
        final_model.save(model_path)
        print(f"  Model saved: {model_path}")
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved: {scaler_path}")
    except Exception as e:
        print(f"Error during saving: {e}")

def main():
    model_run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"--- Starting new training run --- \nRun ID: {model_run_id}")
    X, y_coral, y_scalar, training_cols = load_and_prepare_data_nn(config.FINAL_TRAINING_DATA_PATH)
    if X is not None and not X.empty:
        train_evaluate_save_nn_model(X, y_coral, y_scalar, training_cols, model_run_id)
    else:
        print("Training interrupted due to data loading errors or empty dataset.")

if __name__ == "__main__":
    main()