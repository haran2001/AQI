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
import json
import os
from datetime import datetime
import shutil
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_results_folder(config_name='config'):
    """Create a timestamped results folder for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results_{config_name}_{timestamp}"

    # Create results directory
    os.makedirs(results_folder, exist_ok=True)

    print(f"✓ Created results folder: {results_folder}")
    return results_folder

def save_results_summary(results_folder, config, config_path, dataset_info, model_metrics, training_history, feature_names):
    """Save comprehensive results summary to JSON file"""

    # Extract final epoch metrics
    final_epoch = len(training_history.history['loss']) - 1
    training_metrics = {
        'final_epoch': final_epoch,
        'final_loss': float(training_history.history['loss'][-1]),
        'final_mae': float(training_history.history['mae'][-1]) if 'mae' in training_history.history else None,
        'final_val_loss': float(training_history.history['val_loss'][-1]) if 'val_loss' in training_history.history else None,
        'final_val_mae': float(training_history.history['val_mae'][-1]) if 'val_mae' in training_history.history else None,
        'total_epochs_trained': final_epoch + 1
    }

    # Create comprehensive results summary
    results_summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'config_file_used': config_path,
            'results_folder': results_folder,
            'dataset_name': dataset_info['dataset_name'],
            'total_records': dataset_info['total_records'],
            'clean_records': dataset_info['clean_records'],
            'features_used': feature_names
        },
        'configuration': config,
        'dataset_statistics': dataset_info['statistics'],
        'data_splits': {
            'train_samples': dataset_info['train_samples'],
            'test_samples': dataset_info['test_samples'],
            'validation_samples': dataset_info.get('validation_samples', 'Not specified')
        },
        'model_performance': {
            'test_metrics': model_metrics,
            'training_metrics': training_metrics
        },
        'model_architecture': {
            'total_parameters': dataset_info.get('total_parameters', 'Not calculated'),
            'trainable_parameters': dataset_info.get('trainable_parameters', 'Not calculated'),
            'layers': [layer_config for layer_config in config['model']['layers']]
        }
    }

    # Save to JSON file
    results_file = os.path.join(results_folder, 'results_summary.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"✓ Results summary saved to {results_file}")
    return results_file

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
    statistics = {}
    for i, feature in enumerate(data_config['features']):
        feature_stats = {
            'mean': float(X[:, i].mean()),
            'min': float(X[:, i].min()),
            'max': float(X[:, i].max()),
            'std': float(X[:, i].std())
        }
        statistics[feature] = feature_stats
        print(f"{feature}: Mean={feature_stats['mean']:.1f}, Min={feature_stats['min']:.1f}, Max={feature_stats['max']:.1f}")

    target_stats = {
        'mean': float(y.mean()),
        'min': float(y.min()),
        'max': float(y.max()),
        'std': float(y.std())
    }
    statistics[data_config['target']] = target_stats
    print(f"{data_config['target']}: Mean={target_stats['mean']:.1f}, Min={target_stats['min']:.1f}, Max={target_stats['max']:.1f}")

    # Create dataset info
    dataset_info = {
        'dataset_name': data_config['input_file'],
        'total_records': len(df),
        'clean_records': len(df_clean),
        'statistics': statistics
    }

    return X, y, df_clean, dataset_info

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

def plot_results(config, history, y_test, y_pred, X_test, scaler_X, feature_names, results_folder=None):
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

    # Save to results folder if provided, otherwise use config path
    if results_folder:
        plots_path = os.path.join(results_folder, 'model_performance_plots.png')
    else:
        plots_path = output_config['plots_path']

    plt.savefig(plots_path, dpi=output_config['plot_dpi'], bbox_inches='tight')
    plt.show()
    print(f"\n✓ Plots saved to {plots_path}")

def save_model_and_predictions(config, model, scaler_X, scaler_y, y_test, y_pred, X_test, feature_names, results_folder=None):
    """Save model and predictions"""
    output_config = config['output']

    # Determine save paths
    if results_folder:
        model_path = os.path.join(results_folder, 'model.keras')
        scaler_x_path = os.path.join(results_folder, 'scaler_X.pkl')
        scaler_y_path = os.path.join(results_folder, 'scaler_y.pkl')
        predictions_path = os.path.join(results_folder, 'predictions.csv')
    else:
        model_path = output_config['model_path']
        scaler_x_path = output_config['scaler_x_path']
        scaler_y_path = output_config['scaler_y_path']
        predictions_path = output_config['predictions_path']

    # Save model
    if output_config['save_model']:
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")

    # Save scalers
    if output_config['save_scalers']:
        import joblib
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        print(f"✓ Scalers saved to {scaler_x_path} and {scaler_y_path}")

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
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved to {predictions_path}")

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

    # Create results folder
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    results_folder = create_results_folder(config_name)

    # Copy config file to results folder
    config_copy_path = os.path.join(results_folder, 'config_used.yaml')
    shutil.copy2(config_path, config_copy_path)
    print(f"✓ Configuration file copied to {config_copy_path}")

    print("="*60)
    print("AQI PREDICTION USING DEEP NEURAL NETWORK")
    print("="*60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Results will be saved to: {results_folder}")
    print(f"Independent Variables: {config['data']['features']}")
    print(f"Dependent Variable: {config['data']['target']}")
    print(f"Train-Test Split: {100*(1-config['data']['test_split']):.0f}-{100*config['data']['test_split']:.0f}")
    print("="*60)

    # Load and prepare data
    print("\n1. LOADING AND PREPARING DATA")
    print("-"*40)

    try:
        X, y, df, dataset_info = load_and_prepare_data(config)
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

    # Update dataset info with splits
    dataset_info['train_samples'] = len(X_train)
    dataset_info['test_samples'] = len(X_test)

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

    # Update dataset info with validation split
    dataset_info['validation_samples'] = len(X_val_scaled)

    # Train model
    print("\n2. TRAINING DNN MODEL")
    print("-"*40)
    model, history = train_model(config, X_train_scaled, y_train_scaled,
                                 X_val_scaled, y_val_scaled)

    # Add model parameter counts
    dataset_info['total_parameters'] = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    dataset_info['trainable_parameters'] = trainable_params

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

    # Create model metrics dict
    model_metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }

    # Add MAPE if applicable
    if y_test.min() > 0:
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        model_metrics['mape'] = float(mape)

    # Plot results
    print("\n4. PLOTTING RESULTS")
    print("-"*40)
    plot_results(config, history, y_test, y_pred, X_test_scaled, scaler_X, feature_names, results_folder)

    # Save model and predictions
    print("\n5. SAVING MODEL AND PREDICTIONS")
    print("-"*40)
    save_model_and_predictions(config, model, scaler_X, scaler_y, y_test, y_pred,
                             X_test_scaled, feature_names, results_folder)

    # Save comprehensive results summary
    print("\n6. SAVING RESULTS SUMMARY")
    print("-"*40)
    save_results_summary(results_folder, config, config_path, dataset_info,
                        model_metrics, history, feature_names)

    # Example predictions
    if config['evaluation']['show_sample_predictions']:
        print("\n7. EXAMPLE PREDICTIONS")
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
    print(f"All results saved to: {results_folder}")
    print("Contents:")
    print(f"  - config_used.yaml: Configuration file used for this run")
    print(f"  - results_summary.json: Comprehensive results and metrics")
    print(f"  - model_performance_plots.png: Training and evaluation plots")
    print(f"  - model.keras: Trained model")
    print(f"  - scaler_X.pkl, scaler_y.pkl: Feature and target scalers")
    print(f"  - predictions.csv: Test set predictions vs actual values")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DNN model for AQI prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')

    args = parser.parse_args()
    main(args.config)