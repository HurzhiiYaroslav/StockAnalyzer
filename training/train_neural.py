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
from .feature_analyzer import analyze_feature_importance
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)

def to_coral_format(y_scalar, num_classes):
    coral_y = np.zeros((len(y_scalar), num_classes - 1), dtype=int)
    for i, rank in enumerate(y_scalar):
        if 1 <= rank <= num_classes:
            coral_y[i, :rank - 1] = 1 
    return coral_y

def coral_logits_to_probs(logits):
    return tf.math.sigmoid(logits)

def coral_probs_to_rank(probs):
    return tf.reduce_sum(tf.cast(probs > 0.5, dtype=tf.int32), axis=1) + 1

def load_and_prepare_data_nn(file_path, num_classes_for_coral):
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_csv(file_path, index_col='date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None, None, None
    
    print(f"Loaded {len(data)} rows.")
    if 'Rating' not in data.columns:
        print("Error: 'Rating' column is missing.")
        return None, None, None, None
    
    if 'ticker' in data.columns:
        data_for_X = data.drop(columns=['ticker', 'Rating'])
    else:
        data_for_X = data.drop(columns=['Rating'])

    data_for_X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_scalar_temp = data['Rating'].astype(int)
    combined_for_dropna = pd.concat([data_for_X, y_scalar_temp], axis=1)
    combined_for_dropna.dropna(inplace=True)
    
    print(f"Data after cleaning (NaN, inf): {len(combined_for_dropna)} rows.")

    if combined_for_dropna.empty:
        print("Dataset is empty after cleaning.")
        return None, None, None, None

    X = combined_for_dropna.drop(columns=['Rating'])
    y_scalar = combined_for_dropna['Rating']

    if X.empty or y_scalar.empty:
        print("X or y arrays are empty.")
        return None, None, None, None
        
    training_columns = X.columns.tolist()
    y_coral = to_coral_format(y_scalar.values, num_classes_for_coral)
    
    return X, y_coral, y_scalar, training_columns

def build_model_for_tuner(hp, input_shape, num_classes):
    num_outputs = num_classes - 1
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=5, step=1)

    for i in range(num_hidden_layers):
        with hp.conditional_scope("num_hidden_layers", list(range(1, 6))):
            if i < num_hidden_layers:
                model.add(layers.Dense(
                    units=hp.Int(f'units_layer_{i}', min_value=32, max_value=768, step=32),
                    activation=hp.Choice(f'activation_layer_{i}', values=['relu', 'elu', 'tanh', 'selu', 'gelu']),
                    kernel_initializer=hp.Choice(f'initializer_layer_{i}', values=['he_normal', 'glorot_uniform', 'lecun_normal']),
                    kernel_regularizer=keras.regularizers.l2(
                        hp.Float(f'l2_kernel_layer_{i}', min_value=1e-5, max_value=1e-2, sampling='log')
                    )
                ))
                if hp.Boolean(f'batch_norm_layer_{i}', default=False):
                    model.add(layers.BatchNormalization())
                
                model.add(layers.Dropout(
                    rate=hp.Float(f'dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.05)
                ))

    model.add(layers.Dense(num_outputs)) 

    optimizer_choice = hp.Choice('optimizer', ['adam', 'adamw', 'nadam', 'rmsprop'])
    
    use_lr_schedule = hp.Boolean('use_lr_schedule', default=False)
    
    if use_lr_schedule:
        initial_learning_rate = hp.Float("initial_learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
        decay_steps = 1000
        decay_rate = hp.Float("decay_rate", min_value=0.85, max_value=0.99)
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
        )
    else:
        learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate,
                                           weight_decay=hp.Float('weight_decay_adamw', min_value=1e-5, max_value=1e-2, sampling='log'))
    elif optimizer_choice == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate,
                                             rho=hp.Float('rmsprop_rho', min_value=0.8, max_value=0.99, default=0.9))
                                             
    model.compile(optimizer=optimizer, loss=coral_loss, metrics=['accuracy'])
    return model

def coral_loss(y_true_coral, y_pred_logits):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true_coral, y_pred_logits, from_logits=True)
    )

def train_evaluate_save_nn_model(X_df, y_coral, y_scalar_for_eval, training_columns, num_classes_for_coral):
    if X_df is None or y_coral is None or y_scalar_for_eval is None or training_columns is None:
        return

    print("Splitting data...")
    try:
        X_train_val, X_test_df, y_train_val_coral, y_test_coral, y_train_val_scalar, y_test_scalar = train_test_split(
            X_df, y_coral, y_scalar_for_eval, test_size=0.2, random_state=4, stratify=y_scalar_for_eval
        )
        X_train_df, X_val_df, y_train_coral, y_val_coral, y_train_scalar, y_val_scalar = train_test_split(
            X_train_val, y_train_val_coral, y_train_val_scalar, test_size=0.2, random_state=4, stratify=y_train_val_scalar
        )
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        return

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    print("Searching for hyperparameters with Keras Tuner...")
    input_shape_for_tuner = (X_train_scaled.shape[1],)
    
    def model_builder_wrapper(hp):
        return build_model_for_tuner(hp, input_shape=input_shape_for_tuner, num_classes=num_classes_for_coral)

    tuner = kt.Hyperband(
        model_builder_wrapper,
        objective=kt.Objective("val_loss", direction="min"),
        max_epochs=15,
        factor=3,
        directory='training/keras_tuner_dir_stock',
        project_name='stock_coral_hyperband'
    )
    stop_early_tuner = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    BATCH_SIZE = 512
    tuner_train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_coral))
    tuner_train_ds = tuner_train_ds.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    tuner_val_ds = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_coral))
    tuner_val_ds = tuner_val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    tuner.search(tuner_train_ds,
                 epochs=30,
                 validation_data=tuner_val_ds,
                 callbacks=[stop_early_tuner],
                 verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    print(f"  Number of layers: {best_hps.get('num_hidden_layers')}")
    print(f"  Learning rate: {best_hps.get('learning_rate'):.5f}")
    print(f"  Optimizer: {best_hps.get('optimizer')}")
    for i in range(best_hps.get('num_hidden_layers')):
        print(f"  Layer {i}: Units: {best_hps.get(f'units_layer_{i}')}, Activation: {best_hps.get(f'activation_layer_{i}')}, L2: {best_hps.get(f'l2_kernel_layer_{i}')}, Dropout: {best_hps.get(f'dropout_layer_{i}')}, Batch Norm: {best_hps.get(f'batch_norm_layer_{i}')}")
    
    print("\nBuilding and training the final model...")
    final_model = tuner.hypermodel.build(best_hps)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_coral))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_coral))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    early_stopping_final = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = final_model.fit(train_dataset,
                              epochs=50, 
                              validation_data=val_dataset,
                              callbacks=[early_stopping_final],
                              verbose=1)
    print("Final training complete.")

    print("\nEvaluating the final model...")
    test_logits = final_model.predict(X_test_scaled, verbose=0)
    test_probs = coral_logits_to_probs(test_logits)
    y_pred_scalar_nn = coral_probs_to_rank(test_probs).numpy()

    mae_nn = mean_absolute_error(y_test_scalar, y_pred_scalar_nn)
    accuracy_exact_nn = accuracy_score(y_test_scalar, y_pred_scalar_nn)
    within_one_accuracy_nn = np.mean(np.abs(y_test_scalar.values - y_pred_scalar_nn) <= 1)
    
    print(f"  NN MAE: {mae_nn:.4f}")
    print(f"  NN Accuracy (exact): {accuracy_exact_nn:.4f}")
    print(f"  NN Accuracy (+/- 1 score): {within_one_accuracy_nn:.4f}")

    print("  NN Confusion Matrix:")
    labels_nn = sorted(list(set(y_test_scalar.unique()) | set(np.unique(y_pred_scalar_nn))))
    if not labels_nn:
        print("    No labels for confusion matrix.")
    else:
        try:
            cm_nn = confusion_matrix(y_test_scalar, y_pred_scalar_nn, labels=labels_nn)
            cm_df_nn = pd.DataFrame(cm_nn, index=[f"{i}" for i in labels_nn], columns=[f"{i}" for i in labels_nn])
            print(cm_df_nn)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df_nn, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix for Neural Network', fontsize=20 )
            plt.xlabel('Predicted Values', fontsize=18)
            plt.ylabel('True Values', fontsize=16)
            plt.tight_layout()
            
            plot_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix_nn.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"    Confusion matrix saved to: {plot_path}")
            
        except ValueError as e:
            print(f"    Failed to build confusion matrix: {e}")

    print("\nSaving model and components...")
    try:
        final_model.save(config.MODEL_PATH)
        print(f"  Model saved: {config.MODEL_PATH}")
        joblib.dump(scaler, config.SCALER_PATH)
        print(f"  Scaler saved: {config.SCALER_PATH}")
        joblib.dump(training_columns, config.COLUMNS_PATH)
        print(f"  Columns saved: {config.COLUMNS_PATH}")
    except Exception as e:
        print(f"Error during saving: {e}")

    print("\nStarting feature importance analysis...")
    analyze_feature_importance(final_model, X_test_scaled, y_test_scalar, training_columns, model_type="keras_coral")

def main():
    X, y_coral, y_scalar, training_cols = load_and_prepare_data_nn(config.FINAL_TRAINING_DATA_PATH, config.NUM_CLASSES)
    if X is not None and y_coral is not None and y_scalar is not None and training_cols is not None:
        if not X.empty and not y_scalar.empty:
            train_evaluate_save_nn_model(X, y_coral, y_scalar, training_cols, config.NUM_CLASSES)
        else:
            print("X or y are empty after loading.")
    else:
        print("Training interrupted due to data loading errors.")

if __name__ == "__main__":
    main()