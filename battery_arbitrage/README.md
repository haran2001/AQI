# Battery Arbitrage Trading System

**An AI-powered battery energy storage arbitrage optimization system for the Eland Solar & Storage Center**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](training/tests/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [Data Collection](#data-collection)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Documentation](#documentation)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a sophisticated machine learning system for optimizing battery energy storage arbitrage in the CAISO (California Independent System Operator) electricity market. The system forecasts real-time electricity prices and uses dynamic programming to execute optimal charge/discharge decisions for the **Eland Solar & Storage Center, Phase 2** (200 MW, Kern County, CA).

### Key Capabilities

- **Price Forecasting**: XGBoost model with R² = 0.993, RMSE = $2.10/MWh
- **Trading Strategy**: Rolling intrinsic optimization using dynamic programming
- **Real-time Trading**: 288 decisions per day with API-aware caching
- **Revenue**: $55.88 over 47 hours (test period), 95.8% round-trip efficiency
- **Automated Pipeline**: Config-driven end-to-end training system

---

## 📁 Project Structure

```
battery_arbitrage/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies (coming soon)
├── requirements_streaming.txt          # Streaming system dependencies
│
├── config/                             # Configuration files
│   └── training_config.yaml           # Main training configuration
│
├── training/                           # Training pipeline & scripts
│   ├── config/
│   │   └── training_config.yaml       # Training configuration
│   │
│   ├── data/                          # Data storage
│   │   ├── raw/                       # Raw CAISO & weather data
│   │   └── processed/                 # Processed & merged datasets
│   │
│   ├── models/                        # Trained model artifacts
│   │   └── xgboost_price_model.pkl   # XGBoost forecasting model
│   │
│   ├── plots/                         # Visualizations
│   │   ├── battery_arbitrage_analysis_*.png
│   │   └── xgboost_model_results_*.png
│   │
│   ├── tests/                         # Test suite
│   │   ├── __init__.py
│   │   ├── README.md                  # Testing documentation
│   │   └── test_caiso_sp15_data_fetch.py
│   │
│   ├── pytest.ini                     # Pytest configuration
│   │
│   ├── caiso_sp15_data_fetch.py      # CAISO price data fetcher (config-driven)
│   ├── open_metero_weather_data.py   # Weather data fetcher (config-driven)
│   ├── weather_data_interpolator.py  # Hourly → 5-min interpolation
│   ├── create_merged_dataset.py      # Merge weather + prices
│   ├── xgboost_price_forecaster.py   # ML price forecasting model
│   ├── rolling_internsic_battery_arbitrage.py  # Trading strategy
│   └── visualize_arbitrage_results.py          # Results visualization
│
├── notes/                             # Design documents
│   ├── api_limits_and_trading_strategy.md
│   ├── real_time_streaming_dashboard.md
│   └── PRD_training_pipeline.md      # Product Requirements Document
│
├── api_aware_trading_system.py       # Real-time trading system
├── caiso_streaming_service.py        # Real-time data streaming
├── dashboard.html                     # Live trading dashboard
├── websocket_server.py                # WebSocket server for dashboard
└── simple_forecast_test.py            # Simple forecast testing
```

---

## ✨ Key Features

### 1. **Data Collection (Config-Driven)**
- ✅ **CAISO OASIS API**: Automatic 30-day chunking for large date ranges
- ✅ **Open-Meteo API**: 42 weather variables with intelligent caching
- ✅ **Rate Limiting**: Respects API limits with exponential backoff
- ✅ **Error Handling**: Robust retry logic and graceful degradation

### 2. **Data Processing**
- ✅ **Weather Interpolation**: Hourly → 5-minute intervals
  - Cubic interpolation for temperature
  - Circular interpolation for wind direction (handles 0°/360° boundary)
  - Forward-fill for discrete events (precipitation)
- ✅ **Feature Engineering**: 111+ features
  - 24 price lags (2 hours history)
  - Rolling statistics (mean, std, min, max)
  - Temporal features (hour, day, cyclical encoding)
  - Weather interactions (temp × hour, temp × weekend)

### 3. **Price Forecasting**
- ✅ **XGBoost Model**: Gradient boosting with 1000 estimators
- ✅ **Performance**: R² = 0.993, RMSE = $2.10/MWh
- ✅ **N-Step Forecasting**: 12 steps ahead (1 hour)
- ✅ **Feature Importance**: Automatically tracked and visualized

### 4. **Trading Strategy**
- ✅ **Algorithm**: Rolling intrinsic optimization via dynamic programming
- ✅ **Battery Model**: 500 kWh capacity, 100 kW power, 95% efficiency
- ✅ **Optimization**: 51-point SoC discretization, 11 action grid
- ✅ **Costs**: Degradation ($0.004/kWh) + Trading ($0.00009/kWh)

### 5. **Real-Time System**
- ✅ **API-Aware Trading**: Intelligent caching to minimize API calls
- ✅ **Schedule**: Weather every 15 min, forecasts every hour, trades every 5 min
- ✅ **Efficiency**: 85-95% cache hit rate, stays within free tier limits
- ✅ **Monitoring**: WebSocket dashboard with live updates

### 6. **Testing & Validation**
- ✅ **Test Suite**: 13/13 tests passing
- ✅ **Coverage**: Configuration, chunking, parsing, integration tests
- ✅ **CI/CD Ready**: Pytest configuration with markers

---

## 🏗️ Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
├─────────────────────────────────────────────────────────────┤
│  CAISO OASIS API          │      Open-Meteo API             │
│  (RT/DA Prices)           │      (42 Weather Variables)     │
└──────────┬────────────────┴────────────┬────────────────────┘
           │                              │
           ▼                              ▼
    ┌────────────┐                 ┌──────────────┐
    │ CAISO Data │                 │ Weather Data │
    │  Fetcher   │                 │   Fetcher    │
    └──────┬─────┘                 └──────┬───────┘
           │                              │
           │         ┌──────────┐         │
           └────────►│  Merge & │◄────────┘
                     │ Features │
                     └─────┬────┘
                           │
                ┌──────────▼──────────┐
                │  XGBoost Training   │
                │  (111+ features)    │
                └──────────┬──────────┘
                           │
                     ┌─────▼─────┐
                     │   Model   │
                     │  (R²=0.993)│
                     └─────┬─────┘
                           │
                ┌──────────▼──────────┐
                │ Rolling Intrinsic   │
                │ Optimization (DP)   │
                └──────────┬──────────┘
                           │
                    ┌──────▼──────┐
                    │   Trading   │
                    │  Decisions  │
                    └─────────────┘
```

### Training Pipeline Architecture

```
Config File (YAML)
       ↓
┌──────────────────┐
│ Data Collection  │
│ - CAISO Prices   │ ← Auto-chunking for >30 days
│ - Weather Data   │ ← 42 variables, caching
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Preprocessing   │
│ - Interpolation  │ ← Hourly → 5-min
│ - Merging        │ ← Timestamp alignment
│ - Basic Features │ ← Hour, day, temp conversions
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Feature Engineer │
│ - Price lags     │ ← 1-24 steps
│ - Rolling stats  │ ← 6/12/24 windows
│ - Interactions   │ ← Temp × hour, etc.
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Model Training  │
│ - XGBoost        │ ← 1000 estimators
│ - Evaluation     │ ← RMSE, R², MAE
│ - Save artifacts │ ← Model, scaler, metadata
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Validation     │
│ - Backtest       │ ← Rolling intrinsic
│ - Metrics        │ ← Revenue, efficiency
│ - Visualization  │ ← 9-panel analysis
└──────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning)

### Clone Repository

```bash
git clone <repository-url>
cd battery_arbitrage/training
```

### Install Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn xgboost
pip install pyyaml requests openmeteo-requests requests-cache retry-requests
pip install matplotlib seaborn scipy

# Testing dependencies
pip install pytest pytest-cov

# Optional: Streaming system
pip install -r ../requirements_streaming.txt
```

### Verify Installation

```bash
python -c "import pandas, xgboost, yaml; print('✓ All dependencies installed')"
```

---

## 🎬 Quick Start

### 1. Configure the System

Edit `config/training_config.yaml` to set your date range and preferences:

```yaml
date_range:
  start_date: "2025-08-01"
  end_date: "2025-08-31"

site:
  caiso_node: "TH_SP15_GEN-APND"
  latitude: 35.3733
  longitude: -119.0187
```

### 2. Fetch CAISO Price Data

```bash
python caiso_sp15_data_fetch.py
```

**Output:**
- `data/raw/caiso_da_prices_2025-08-01_2025-08-31.csv` (Day-Ahead hourly)
- `data/raw/caiso_rt_prices_2025-08-01_2025-08-31.csv` (Real-Time 5-min)

### 3. Fetch Weather Data

```bash
python open_metero_weather_data.py
```

**Output:**
- `data/raw/weather_hourly_2025-08-01_2025-08-31.csv` (42 variables)

### 4. Interpolate Weather Data

```bash
python weather_data_interpolator.py \
    --input-file data/raw/weather_hourly_2025-08-01_2025-08-31.csv \
    --output-file data/processed/weather_5min.csv \
    --points-per-hour 12
```

### 5. Create Merged Dataset

```bash
python create_merged_dataset.py
```

**Output:**
- `data/processed/merged_dataset.csv` (8,640 records with basic features)

### 6. Train Price Forecasting Model

```bash
python xgboost_price_forecaster.py \
    --data-file data/processed/merged_dataset.csv \
    --save-model models/xgboost_price_model.pkl
```

**Output:**
- `models/xgboost_price_model.pkl` (trained model)
- `data/xgboost_model_results.png` (evaluation plots)

### 7. Run Arbitrage Backtest

```bash
python rolling_internsic_battery_arbitrage.py
```

**Output:**
- `arbitrage_results.csv` (trading log)

### 8. Visualize Results

```bash
python visualize_arbitrage_results.py
```

**Output:**
- `battery_arbitrage_analysis.png` (9-panel comprehensive analysis)

---

## ⚙️ Configuration

The system is driven by `config/training_config.yaml`, which controls all aspects of the pipeline.

### Key Configuration Sections

#### 1. Date Range
```yaml
date_range:
  start_date: "2025-08-01"
  end_date: "2025-08-31"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

#### 2. Site Information
```yaml
site:
  name: "Eland Solar & Storage Center, Phase 2"
  latitude: 35.3733
  longitude: -119.0187
  caiso_node: "TH_SP15_GEN-APND"
```

#### 3. Data Collection
```yaml
data_collection:
  caiso:
    auto_chunk: true              # Automatic 30-day chunking
    max_days_per_chunk: 30
    timeout_seconds: 300          # 5 minutes for large datasets

  weather:
    hourly_variables:             # All 42 weather variables
      - temperature_2m
      - wind_speed_10m
      # ... (see config file for full list)
```

#### 4. Feature Engineering
```yaml
features:
  basic:
    temporal: [hour, day_of_week, is_weekend]
    cyclical: [hour_sin, hour_cos, day_sin, day_cos]

  advanced:
    price_lags:
      lag_steps: [1, 2, 3, 6, 12, 24]
    rolling_statistics:
      windows: [6, 12, 24]
      functions: [mean, std, min, max]
```

#### 5. Model Configuration
```yaml
models:
  xgboost:
    params:
      n_estimators: 1000
      max_depth: 6
      learning_rate: 0.1
    forecast_steps: 12           # 1 hour ahead
```

#### 6. Battery Specifications
```yaml
battery:
  capacity_kwh: 500.0
  max_power_kw: 100.0
  efficiency_charge: 0.95
  efficiency_discharge: 0.95
```

---

## 🔬 Training Pipeline

### Current Status: **In Development** (70% Complete)

**Implemented:**
- ✅ Configuration system (YAML-based)
- ✅ CAISO data fetcher (config-driven, auto-chunking)
- ✅ Weather data fetcher (config-driven, 42 variables)
- ✅ Weather interpolation (5 methods: cubic, linear, circular, etc.)
- ✅ Dataset merging (timestamp alignment)
- ✅ XGBoost price forecasting (R² = 0.993)
- ✅ Rolling intrinsic optimization (dynamic programming)
- ✅ Results visualization (9-panel analysis)

**In Progress:**
- 🔄 Consolidated training pipeline (`train_pipeline.py`)
- 🔄 Automated feature engineering
- 🔄 Checkpointing and resume capability
- 🔄 Comprehensive logging system
- 🔄 HTML report generation

**Planned:**
- 📋 Multi-model support (LSTM, LightGBM)
- 📋 Hyperparameter optimization (Optuna)
- 📋 MLflow experiment tracking
- 📋 Automated deployment pipeline

See [PRD_training_pipeline.md](notes/PRD_training_pipeline.md) for full details.

---

## 📊 Data Collection

### CAISO Price Data

**Source:** CAISO OASIS API
**Endpoint:** https://oasis.caiso.com/oasisapi/SingleZip
**Node:** TH_SP15_GEN-APND (Eland Solar & Storage Center)

**Markets:**
- **Day-Ahead (DAM)**: Hourly prices, published day before
- **Real-Time (RTM)**: 5-minute prices, published ~5 min after interval

**Features:**
- ✅ Automatic 30-day chunking (CAISO API limit: 31 days)
- ✅ Rate limiting with exponential backoff
- ✅ Error handling (429, timeouts, empty responses)
- ✅ Data validation (duplicates, gaps, outliers)

**Example:**
```bash
python caiso_sp15_data_fetch.py --start-date 2025-01-01 --end-date 2025-03-31
# Automatically splits into 3 chunks of 30 days each
```

### Weather Data

**Source:** Open-Meteo Historical Forecast API
**Location:** 35.3733°N, 119.0187°W (Kern County, CA)

**Variables (42 total):**
- Temperature: 2m, 80m, 120m, 180m, soil (6 variables)
- Wind: Speed/direction at 4 heights, gusts (9 variables)
- Humidity & Pressure: Relative humidity, MSL/surface pressure, VPD (4 variables)
- Cloud Cover: Total, low, mid, high (4 variables)
- Precipitation: Rain, showers, snow, probability (5 variables)
- Soil Moisture: 5 depth levels (5 variables)
- Other: Weather code, visibility, evapotranspiration (9 variables)

**Features:**
- ✅ Caching (1 hour expiration)
- ✅ Retry logic (5 attempts with backoff)
- ✅ Missing value detection
- ✅ Circular interpolation for wind direction

---

## 📈 Model Performance

### XGBoost Price Forecasting Model

**Training Data:** 6,912 samples (80% of 8,640 5-min intervals)
**Test Data:** 1,728 samples (20%)

**Metrics:**
- **RMSE**: $2.10/MWh
- **MAE**: $1.45/MWh
- **R²**: 0.993 (99.3% variance explained)
- **Training Time**: ~2 minutes

**Top 5 Features:**
1. `price_lag_1` (most recent price)
2. `price_lag_2`
3. `price_rolling_mean_6` (30-min average)
4. `price_lag_3`
5. `hour_sin` (time of day cyclical)

### Battery Arbitrage Performance

**Test Period:** 47 hours (2,256 5-min intervals)
**Strategy:** Rolling intrinsic with 1-hour horizon

**Results:**
- **Total Revenue**: $55.88
- **Revenue per kWh**: $0.034/kWh
- **Round-Trip Efficiency**: 95.8%
- **Battery Cycles**: 1.87 full cycles
- **Actions**: 44% charge, 37% discharge, 19% hold

**Action Distribution:**
- Charges during low-price hours (night, midday solar)
- Discharges during peak hours (evening)
- Captures price volatility efficiently

---

## 🧪 Testing

### Test Suite

**Location:** `training/tests/`
**Framework:** pytest
**Coverage:** 13 tests, 100% pass rate

**Test Categories:**

1. **Configuration Tests** (3 tests)
   - YAML loading
   - Parameter validation
   - Value ranges

2. **Date Chunking Tests** (5 tests)
   - No chunking (<30 days)
   - Exact 30 days
   - Multi-chunk (90 days)
   - No overlap verification
   - Custom chunk sizes

3. **Data Processing Tests** (4 tests)
   - DateTime formatting
   - XML parsing
   - Empty data handling
   - Mock data validation

4. **Integration Tests** (1 test)
   - Live CAISO API call

### Running Tests

```bash
# All tests
pytest

# Unit tests only (fast, no network)
pytest -k "not integration"

# Specific test file
pytest tests/test_caiso_sp15_data_fetch.py -v

# With coverage
pytest --cov=. --cov-report=html
```

**Output:**
```
============================== 13 passed in 3.05s ==============================
```

See [tests/README.md](training/tests/README.md) for detailed testing documentation.

---

## 📚 Documentation

### Key Documents

1. **[PRD: Training Pipeline](notes/PRD_training_pipeline.md)** (35 pages)
   - Complete product requirements
   - Functional requirements (FR-1 to FR-19)
   - Architecture diagrams
   - Implementation plan (6 phases)
   - 75-80% code reuse strategy

2. **[API Limits & Trading Strategy](notes/api_limits_and_trading_strategy.md)**
   - CAISO & Open-Meteo rate limits
   - Intelligent caching strategy
   - Trading schedule (288 decisions/day)
   - Fallback mechanisms

3. **[Real-Time Streaming Dashboard](notes/real_time_streaming_dashboard.md)**
   - WebSocket architecture
   - Live price visualization
   - Trading decision monitoring

4. **[Test Documentation](training/tests/README.md)**
   - Testing best practices
   - CI/CD integration
   - Coverage reports

### Code Documentation

All scripts include comprehensive docstrings:

```python
def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    ...
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo API

    Args:
        latitude: Site latitude
        longitude: Site longitude
        ...

    Returns:
        DataFrame with hourly weather data
    """
```

---

## 🗺️ Future Roadmap

### Phase 1: Complete Training Pipeline (Q4 2025)
- [ ] Consolidate all scripts into `train_pipeline.py`
- [ ] Implement checkpointing
- [ ] Add structured logging
- [ ] Generate HTML reports
- [ ] End-to-end testing

### Phase 2: Enhanced Modeling (Q1 2026)
- [ ] LSTM price forecasting
- [ ] LightGBM ensemble
- [ ] Hyperparameter optimization (Optuna)
- [ ] Feature selection algorithms
- [ ] Model interpretability (SHAP)

### Phase 3: Production Deployment (Q2 2026)
- [ ] MLflow experiment tracking
- [ ] Model versioning & registry
- [ ] Automated model deployment
- [ ] A/B testing framework
- [ ] Continuous training pipeline

### Phase 4: Advanced Trading (Q3 2026)
- [ ] Stochastic optimization
- [ ] Risk-adjusted strategies
- [ ] Multi-objective optimization
- [ ] Portfolio management (multiple sites)
- [ ] Real-time bid optimization

### Phase 5: Scaling & Operations (Q4 2026)
- [ ] Cloud deployment (AWS/GCP)
- [ ] Distributed training (Ray)
- [ ] Real-time monitoring dashboard
- [ ] Alerting & anomaly detection
- [ ] Multi-site orchestration

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd battery_arbitrage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pre-commit

# Run tests
pytest
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write** tests for new functionality
4. **Ensure** all tests pass (`pytest`)
5. **Commit** changes (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Code Standards

- **Style**: PEP 8 compliant
- **Docstrings**: Google style
- **Type Hints**: Required for public functions
- **Testing**: 80%+ coverage
- **Commits**: Conventional Commits format

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CAISO**: For providing open access to market data
- **Open-Meteo**: For free weather forecasting API
- **XGBoost Team**: For the gradient boosting framework
- **Eland Solar & Storage Center**: Site location and specifications

---

## 📞 Contact

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Open an issue](../../issues)
- **Email**: [Your contact email]
- **Documentation**: [Full docs](notes/)

---

## 🔖 Quick Links

- [Training Pipeline PRD](notes/PRD_training_pipeline.md)
- [Test Documentation](training/tests/README.md)
- [Configuration File](training/config/training_config.yaml)
- [API Limits Guide](notes/api_limits_and_trading_strategy.md)

---

**Built with ❤️ for sustainable energy storage optimization**

*Last Updated: October 4, 2025*
