# 🌍 Bangalore Air Quality Index (AQI) Prediction System

A sophisticated Deep Neural Network (DNN) system for predicting Air Quality Index (AQI) in Bangalore using historical weather and pollution data. The project includes automated data collection, configurable model architectures, and comprehensive experiment tracking.

## 📊 Project Overview

This project implements a machine learning pipeline to predict AQI values based on various environmental factors including:
- **Pollutant concentrations** (PM2.5, PM10, CO, NO2, SO2, O3)
- **Weather conditions** (temperature, humidity, wind speed, pressure)
- **Environmental factors** (UV index, cloud cover, precipitation)

### Key Features

- 🔄 **Automated Data Collection**: Fetches historical AQI and weather data from Open-Meteo APIs
- 🧠 **Deep Neural Network Models**: Configurable architectures with hyperparameter optimization
- 📈 **Experiment Tracking**: Automatic results folder creation with comprehensive metrics
- ⚙️ **YAML Configuration**: Easy hyperparameter tuning without code changes
- 📊 **Performance Visualization**: Automatic generation of training curves and prediction plots
- 🎯 **Multiple Configurations**: Pre-built configs targeting different R² scores (baseline, improved, aggressive)

## 🚀 Latest Results

### Model Performance (config_improved_r90)
**Results from run: 2025-09-20 16:55:10**

| Metric | Value |
|--------|-------|
| **R² Score** | **0.733** |
| **MAE** | 9.56 |
| **RMSE** | 11.87 |
| **Training Samples** | 596 |
| **Test Samples** | 106 |

*Significant improvement from baseline R² of 0.14 to 0.73 by adding pollution and weather features!*

## 📁 Project Structure

```
aqi/
├── Data Collection
│   ├── bangalore_historical_data.py    # Fetches 1 year of historical data
│   ├── aqi_data.py                     # Open-Meteo air quality API client
│   ├── weather_data.py                 # Weather data collection script
│   └── test.py                         # WAQI API integration
│
├── Model Training
│   ├── aqi_dnn_predictor.py           # Main DNN training script
│   └── aqi_dnn_predictor_configurable.py # Advanced configurable version
│
├── Configurations
│   ├── config.yaml                     # Baseline configuration (R²: 0.14)
│   ├── config_improved_r90.yaml        # Improved config targeting 90% R²
│   └── config_aggressive_r90.yaml      # Aggressive config with all features
│
├── Data Files
│   ├── bangalore_historical_aqi_weather.csv  # Combined dataset (744 records)
│   ├── bangalore_daily_aqi_weather.csv      # Daily aggregated data
│   └── bangalore_air_quality_data.db        # SQLite database
│
└── Results (Auto-generated)
    └── results_config_YYYYMMDD_HHMMSS/
        ├── config_used.yaml             # Configuration snapshot
        ├── results_summary.json         # Comprehensive metrics
        ├── model_performance_plots.png  # Training/evaluation plots
        ├── model.keras                  # Trained model
        ├── predictions.csv              # Test predictions
        └── scaler_X.pkl, scaler_y.pkl  # Feature scalers
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd aqi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (for WAQI API):
```bash
echo "WAQI_API_TOKEN='your_token_here'" > .env
```

## 📊 Data Collection

### Fetch Historical Data (Recommended)
```bash
# Fetch 1 year of data (default)
python bangalore_historical_data.py

# Fetch custom duration (e.g., 30 days)
python bangalore_historical_data.py 30
```

This creates:
- `bangalore_historical_aqi_weather.csv` - Hourly data
- `bangalore_daily_aqi_weather.csv` - Daily aggregated data

### Available Features (44 total)
- **Pollution**: PM10, PM2.5, CO, NO2, SO2, O3, dust
- **Weather**: Temperature, humidity, dew point, apparent temperature
- **Wind**: Speed (10m/100m), direction, gusts
- **Pressure**: MSL pressure, surface pressure
- **Environmental**: UV index, cloud cover, precipitation, rain
- **Soil**: Temperature, moisture
- **Calculated**: AQI, vapor pressure deficit

## 🎯 Model Training

### Basic Training
```bash
# Using default config
python aqi_dnn_predictor.py

# Using improved configuration (recommended)
python aqi_dnn_predictor.py --config config_improved_r90.yaml

# Using aggressive configuration (maximum features)
python aqi_dnn_predictor.py --config config_aggressive_r90.yaml
```

### Configuration Comparison

| Configuration | Features | R² Score | MAE | Description |
|--------------|----------|----------|-----|-------------|
| **config.yaml** | 2 (temp, humidity) | 0.14 | 16.7 | Baseline model |
| **config_improved_r90.yaml** | 18 (pollution + weather) | 0.73 | 9.6 | Balanced approach |
| **config_aggressive_r90.yaml** | 26 (all available) | TBD | TBD | Maximum features |

## ⚙️ Configuration System

### YAML Configuration Structure
```yaml
data:
  features: ["pm10", "pm2_5", "temperature_2m", ...]  # Input features
  target: "calculated_aqi"                            # Target variable
  test_split: 0.15                                    # 15% for testing

model:
  layers:
    - units: 256
      activation: "relu"
      dropout: 0.1
      batch_norm: true

training:
  epochs: 500
  batch_size: 16
  learning_rate: 0.0005

callbacks:
  early_stopping:
    enabled: true
    patience: 50
```

### Creating Custom Configurations
1. Copy an existing config file
2. Modify hyperparameters as needed
3. Run with `--config your_config.yaml`

## 📈 Results Tracking

Each training run automatically creates a timestamped results folder containing:

### results_summary.json
- Experiment metadata (timestamp, dataset info)
- Complete configuration used
- Dataset statistics for all features
- Model architecture details
- Training metrics (loss, MAE)
- Test performance (R², MAE, RMSE, MAPE)

### Model Artifacts
- **model.keras**: Trained TensorFlow model
- **scaler_X.pkl, scaler_y.pkl**: Feature/target scalers for preprocessing
- **predictions.csv**: Test set predictions vs actual values
- **model_performance_plots.png**: 6-panel visualization including:
  - Training/validation loss curves
  - MAE progression
  - Actual vs predicted scatter plot
  - Residual analysis
  - Feature importance

## 🔬 Performance Analysis

### Current Best Model (config_improved_r90)

**Key Success Factors:**
1. **Feature Engineering**: Added PM2.5, PM10 (direct AQI components)
2. **Extended Feature Set**: 18 features vs original 2
3. **Optimized Architecture**: 5 layers with batch normalization
4. **Better Training**: 500 epochs with early stopping
5. **Reduced Dropout**: 0.1 vs 0.3 (less regularization needed)

**Performance Metrics:**
- **R² Score**: 0.733 (73.3% variance explained)
- **MAE**: 9.56 AQI units
- **RMSE**: 11.87 AQI units
- **Improvement**: 5.2x better than baseline

### Feature Importance
Top contributing features:
1. PM2.5 (highest correlation with AQI)
2. PM10
3. Temperature
4. Humidity
5. Wind speed

## 🎮 Usage Examples

### Making Predictions
```python
from tensorflow import keras
import joblib
import numpy as np

# Load model and scalers
model = keras.models.load_model('results_config_improved_r90_20250920_165510/model.keras')
scaler_X = joblib.load('results_config_improved_r90_20250920_165510/scaler_X.pkl')
scaler_y = joblib.load('results_config_improved_r90_20250920_165510/scaler_y.pkl')

# Prepare input (must match training features)
features = np.array([[
    25.0,  # pm10
    15.0,  # pm2_5
    25.0,  # temperature_2m
    60.0,  # relative_humidity_2m
    # ... other features
]])

# Scale and predict
features_scaled = scaler_X.transform(features)
prediction_scaled = model.predict(features_scaled)
aqi = scaler_y.inverse_transform(prediction_scaled)

print(f"Predicted AQI: {aqi[0][0]:.1f}")
```

### Batch Processing
```python
import pandas as pd

# Load new data
new_data = pd.read_csv('new_weather_data.csv')

# Select required features (from config)
features = ['pm10', 'pm2_5', 'temperature_2m', ...]
X = new_data[features].values

# Predict
X_scaled = scaler_X.transform(X)
predictions_scaled = model.predict(X_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)
```

## 🚧 Future Improvements

### Planned Enhancements
- [ ] **Reinforcement Learning**: Automated hyperparameter optimization
- [ ] **Time Series Features**: Incorporate temporal patterns
- [ ] **Ensemble Models**: Combine multiple architectures
- [ ] **Real-time Predictions**: API endpoint for live predictions
- [ ] **Geographic Expansion**: Support for multiple cities
- [ ] **Feature Selection**: Automatic feature importance ranking
- [ ] **Cross-validation**: K-fold validation for robust evaluation

### Target Performance
- **Goal**: Achieve 90%+ R² score
- **Strategy**: Add remaining pollution features, optimize architecture
- **Timeline**: Next iteration expected to reach 85-90% R²

## 📝 API Reference

### Data Collection Scripts

#### `bangalore_historical_data.py`
```bash
python bangalore_historical_data.py [days_back]
```
- `days_back`: Number of days to fetch (default: 365)
- Creates CSV files with hourly and daily data

#### `aqi_dnn_predictor.py`
```bash
python aqi_dnn_predictor.py [--config CONFIG_FILE]
```
- `--config`: Path to YAML configuration file
- Creates timestamped results folder

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Open-Meteo**: Free weather and air quality APIs
- **WAQI**: World Air Quality Index project
- **TensorFlow**: Deep learning framework

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. AQI predictions should not replace official air quality monitoring systems.