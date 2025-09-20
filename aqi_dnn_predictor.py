import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_and_prepare_data(config):
    """Load and prepare data for training"""
    data_config = config['data']

    try:
        df = pd.read_csv(data_config['input_file'])
        print(f"✓ Data loaded: {len(df)} records")
    except FileNotFoundError:
        print(f"ERROR: File {data_config['input_file']} not found!")
        print("Please run 'python bangalore_historical_data.py' first to generate the data.")
        raise FileNotFoundError(f"Required file {data_config['input_file']} not found")

    # Select relevant columns
    required_columns = data_config['features'] + [data_config['target']]

    # Check if columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"Required columns missing: {missing_cols}")

    # Remove rows with missing values
    df_clean = df[required_columns].dropna()
    print(f"✓ Clean data: {len(df_clean)} records after removing NaN values")

    if len(df_clean) == 0:
        raise ValueError("No valid data after cleaning. Please check the input file.")

    # Prepare features and target
    X = df_clean[data_config['features']].values
    y = df_clean[data_config['target']].values

    # Print data statistics
    print(f"\nData Statistics:")
    for i, feature in enumerate(data_config['features']):
        print(f"{feature}: Mean={X[:, i].mean():.1f}, Min={X[:, i].min():.1f}, Max={X[:, i].max():.1f}")
    print(f"{data_config['target']}: Mean={y.mean():.1f}, Min={y.min():.1f}, Max={y.max():.1f}")

    return X, y, df_clean

def create_dnn_model(config, input_dim):
    """Create a Deep Neural Network model from config"""
    model_config = config['model']
    training_config = config['training']

    model_layers = []

    # Add hidden layers from config
    for i, layer_config in enumerate(model_config['layers']):
        if i == 0:
            # First layer needs input_dim
            model_layers.append(
                layers.Dense(layer_config['units'],
                           activation=layer_config['activation'],
                           input_dim=input_dim)
            )
        else:
            model_layers.append(
                layers.Dense(layer_config['units'],
                           activation=layer_config['activation'])
            )

        # Add batch normalization if specified
        if layer_config.get('batch_norm', False):
            model_layers.append(layers.BatchNormalization())

        # Add dropout if specified
        if 'dropout' in layer_config and layer_config['dropout'] > 0:
            model_layers.append(layers.Dropout(layer_config['dropout']))

    # Add output layer
    output_config = model_config['output']
    model_layers.append(
        layers.Dense(output_config['units'],
                    activation=output_config['activation'])
    )

    # Create sequential model
    model = keras.Sequential(model_layers)

    # Configure optimizer
    optimizer_config = training_config['optimizer']
    if optimizer_config['type'].lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=optimizer_config['learning_rate'])
    elif optimizer_config['type'].lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=optimizer_config['learning_rate'])
    else:
        optimizer = optimizer_config['type']

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=training_config['loss'],
        metrics=training_config['metrics']
    )

    return model

def get_callbacks(config):
    """Create callbacks from config"""
    callbacks_config = config['callbacks']
    callback_list = []

    # Early stopping
    if callbacks_config['early_stopping']['enabled']:
        es_config = callbacks_config['early_stopping']
        callback_list.append(
            callbacks.EarlyStopping(
                monitor=es_config['monitor'],
                patience=es_config['patience'],
                restore_best_weights=es_config['restore_best_weights'],
                verbose=es_config['verbose']
            )
        )

    # Reduce learning rate
    if callbacks_config['reduce_lr']['enabled']:
        rlr_config = callbacks_config['reduce_lr']
        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor=rlr_config['monitor'],
                factor=rlr_config['factor'],
                patience=rlr_config['patience'],
                min_lr=rlr_config['min_lr'],
                verbose=rlr_config['verbose']
            )
        )

    return callback_list

def train_model(config, X_train, y_train, X_val, y_val):
    """Train the DNN model"""
    training_config = config['training']

    # Create model
    model = create_dnn_model(config, X_train.shape[1])

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Get callbacks
    callback_list = get_callbacks(config)

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callback_list,
        verbose=training_config['verbose']
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""

    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Calculate percentage error if configured
    if y_test.min() > 0:  # Avoid division by zero
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return y_pred, mae, mse, rmse, r2

def plot_results(config, history, y_test, y_pred, X_test, scaler_X, feature_names):
    """Plot training history and predictions"""
    output_config = config['output']

    if not output_config['save_plots']:
        return

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Training history - Loss
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Training history - MAE
    plt.subplot(2, 3, 2)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: Actual vs Predicted
    plt.subplot(2, 3, 3)
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI')
    plt.grid(True, alpha=0.3)

    # Plot 4: Residuals
    plt.subplot(2, 3, 4)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted AQI')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    # Plot 5: Histogram of residuals
    plt.subplot(2, 3, 5)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)

    # Plot 6: Feature importance
    plt.subplot(2, 3, 6)
    X_test_original = scaler_X.inverse_transform(X_test)
    correlations = [np.corrcoef(X_test_original[:, i], y_pred)[0, 1]
                   for i in range(len(feature_names))]

    colors = ['green' if c > 0 else 'red' for c in correlations]
    plt.bar(feature_names, np.abs(correlations), color=colors, alpha=0.7)
    plt.ylabel('Absolute Correlation with Predictions')
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_config['plots_path'], dpi=output_config['plot_dpi'], bbox_inches='tight')
    plt.show()
    print(f"\n✓ Plots saved to {output_config['plots_path']}")

def save_model_and_predictions(config, model, scaler_X, scaler_y, y_test, y_pred, X_test, feature_names):
    """Save model and predictions"""
    output_config = config['output']

    # Save model
    if output_config['save_model']:
        model.save(output_config['model_path'])
        print(f"✓ Model saved to {output_config['model_path']}")

    # Save scalers
    if output_config['save_scalers']:
        import joblib
        joblib.dump(scaler_X, output_config['scaler_x_path'])
        joblib.dump(scaler_y, output_config['scaler_y_path'])
        print(f"✓ Scalers saved to {output_config['scaler_x_path']} and {output_config['scaler_y_path']}")

    # Save predictions
    if output_config['save_predictions']:
        X_test_original = scaler_X.inverse_transform(X_test)

        predictions_data = {}
        for i, feature in enumerate(feature_names):
            predictions_data[feature] = X_test_original[:, i]

        predictions_data['actual_aqi'] = y_test
        predictions_data['predicted_aqi'] = y_pred
        predictions_data['error'] = y_test - y_pred
        predictions_data['absolute_error'] = np.abs(y_test - y_pred)

        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(output_config['predictions_path'], index=False)
        print(f"✓ Predictions saved to {output_config['predictions_path']}")

        # Print sample predictions
        print("\nSample Predictions (first 10):")
        print(predictions_df.head(10).to_string())

def predict_aqi(model, scaler_X, scaler_y, features_dict, feature_names):
    """Make a single prediction"""

    # Prepare input in correct order
    input_data = np.array([[features_dict[name] for name in feature_names]])
    input_scaled = scaler_X.transform(input_data)

    # Make prediction
    prediction_scaled = model.predict(input_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]

    return prediction

def main(config_path='config.yaml'):
    """Main function"""

    # Load configuration
    config = load_config(config_path)
    print("="*60)
    print("AQI PREDICTION USING DEEP NEURAL NETWORK")
    print("="*60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Independent Variables: {config['data']['features']}")
    print(f"Dependent Variable: {config['data']['target']}")
    print(f"Train-Test Split: {100*(1-config['data']['test_split']):.0f}-{100*config['data']['test_split']:.0f}")
    print("="*60)

    # Load and prepare data
    print("\n1. LOADING AND PREPARING DATA")
    print("-"*40)

    try:
        X, y, df = load_and_prepare_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        print("\nExiting...")
        return

    # Get feature names
    feature_names = config['data']['features']

    # Split data
    data_config = config['data']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config['test_split'],
        random_state=data_config['random_state'],
        shuffle=True
    )

    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")

    # Scale features
    scaler_X = StandardScaler() if data_config['scale_features'] else None
    scaler_y = StandardScaler() if data_config['scale_target'] else None

    if scaler_X:
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if scaler_y:
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = y_train

    # Create validation set
    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(
        X_train_scaled, y_train_scaled,
        test_size=data_config['validation_split'],
        random_state=data_config['random_state']
    )

    # Train model
    print("\n2. TRAINING DNN MODEL")
    print("-"*40)
    model, history = train_model(config, X_train_scaled, y_train_scaled,
                                 X_val_scaled, y_val_scaled)

    # Evaluate model
    print("\n3. EVALUATING MODEL")
    print("-"*40)

    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()

    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    else:
        y_pred = y_pred_scaled

    # Evaluate
    _, mae, mse, rmse, r2 = evaluate_model(model, X_test_scaled, y_test)

    # Recalculate metrics on original scale
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nMetrics on Original Scale:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    # Plot results
    print("\n4. PLOTTING RESULTS")
    print("-"*40)
    plot_results(config, history, y_test, y_pred, X_test_scaled, scaler_X, feature_names)

    # Save model and predictions
    print("\n5. SAVING MODEL AND PREDICTIONS")
    print("-"*40)
    save_model_and_predictions(config, model, scaler_X, scaler_y, y_test, y_pred,
                             X_test_scaled, feature_names)

    # Example predictions
    if config['evaluation']['show_sample_predictions']:
        print("\n6. EXAMPLE PREDICTIONS")
        print("-"*40)

        test_cases = config['evaluation']['test_cases']
        print("\nTest Predictions:")

        for test_case in test_cases:
            # Create feature dict
            features_dict = {name: test_case.get(name, 0) for name in feature_names}
            pred = predict_aqi(model, scaler_X, scaler_y, features_dict, feature_names)

            # Print prediction
            feature_str = ", ".join([f"{name}={features_dict[name]}" for name in feature_names])
            desc = test_case.get('description', '')
            print(f"  {feature_str} → Predicted AQI: {pred:.1f}  ({desc})")

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DNN model for AQI prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')

    args = parser.parse_args()
    main(args.config)