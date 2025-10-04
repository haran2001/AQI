# Product Requirements Document: Consolidated Training Pipeline

**Version:** 1.0
**Last Updated:** 2025-10-04
**Status:** Draft
**Owner:** Battery Arbitrage Team
**Project:** Eland Solar & Storage Center ML Training Pipeline

---

## Executive Summary

This document outlines the requirements for a consolidated training pipeline that automates the end-to-end process of training machine learning models for battery arbitrage optimization. The pipeline will replace the current manual execution of 7+ separate scripts with a single, configuration-driven system that handles data collection, preprocessing, feature engineering, model training, and validation.

### Key Objectives
- **Automation**: Reduce manual intervention from 7+ script executions to a single command
- **Reproducibility**: Enable exact reproduction of training runs via configuration files
- **Reliability**: Implement robust error handling, retry logic, and checkpointing
- **Flexibility**: Support multiple models, features, and experimental configurations
- **Maintainability**: Create a modular, well-documented codebase

---

## 1. Background & Context

### 1.1 Current State

The existing training workflow requires manual execution of:
1. `caiso_sp15_data_fetch.py` - Fetch CAISO price data
2. `open_metero_weather_data.py` - Fetch weather data
3. `weather_data_interpolator.py` - Interpolate to 5-min intervals
4. `create_merged_dataset.py` - Merge datasets
5. `xgboost_price_forecaster.py` - Train XGBoost model
6. `rolling_internsic_battery_arbitrage.py` - Run arbitrage strategy
7. `visualize_arbitrage_results.py` - Generate visualizations

**Pain Points:**
- Manual coordination of scripts is error-prone
- No standardized configuration management
- Difficult to reproduce exact training conditions
- No automated validation or reporting
- Inconsistent error handling across scripts
- Hard to track experiments and compare results

### 1.2 Target Users

- **Data Scientists**: Running experiments, tuning hyperparameters
- **ML Engineers**: Deploying models, monitoring performance
- **Researchers**: Testing new features, comparing strategies
- **Operations Team**: Scheduled retraining, production deployments

---

## 2. Goals & Success Metrics

### 2.1 Goals

**Primary Goals:**
1. Automate the complete training workflow from raw data to trained models
2. Enable reproducible experiments through configuration management
3. Reduce training setup time from 30+ minutes to <5 minutes
4. Implement comprehensive error handling and recovery mechanisms

**Secondary Goals:**
5. Generate automated training reports and visualizations
6. Support multiple model architectures (XGBoost, LSTM, etc.)
7. Enable hyperparameter optimization workflows
8. Provide clear logging and monitoring capabilities

### 2.2 Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Setup Time | 30+ min | <5 min | Time to start training run |
| Manual Steps | 7+ scripts | 1 command | Number of user actions |
| Reproducibility | Low | 100% | Identical results from config |
| Error Recovery | Manual | Automatic | Checkpoint resume success rate |
| Experiment Tracking | None | Automated | Config file versioning |
| Training Errors | ~20% runs | <5% runs | Failed run percentage |

### 2.3 Non-Goals

- Real-time model deployment (handled by separate deployment pipeline)
- Production inference serving (handled by `api_aware_trading_system.py`)
- Data quality monitoring dashboard (future enhancement)
- Multi-site training orchestration (single-site focus)

---

## 3. Functional Requirements

### 3.1 Configuration Management

**FR-1: YAML/JSON Configuration File**
- **Priority:** P0 (Critical)
- **Description:** Support YAML or JSON configuration files that define all pipeline parameters
- **Requirements:**
  - Date ranges (start, end, train/test split ratios)
  - Site information (location, CAISO node, zone)
  - Data collection settings (API endpoints, retry logic)
  - Feature engineering specifications (lags, rolling windows)
  - Model hyperparameters (per model type)
  - Battery configuration (for validation)
  - Output paths and directories
- **Acceptance Criteria:**
  - Config file loads without errors
  - All parameters are validated before pipeline execution
  - Invalid configs produce clear error messages
  - Support both YAML and JSON formats

**FR-2: Configuration Validation**
- **Priority:** P0
- **Description:** Validate configuration files before pipeline execution
- **Requirements:**
  - Check required fields are present
  - Validate data types and ranges
  - Verify date ranges are logical (start < end)
  - **Calculate and warn if date range >30 days** (will require chunking for CAISO)
  - Check file paths are writable
  - Validate model parameters are within acceptable ranges
  - Estimate total API calls required (weather + CAISO chunks)
- **Acceptance Criteria:**
  - Validation runs in <1 second
  - Clear error messages identify problematic fields
  - Warnings for deprecated or suboptimal settings
  - Display expected number of API chunks for large date ranges

### 3.2 Data Collection

**FR-3: CAISO Price Data Fetching**
- **Priority:** P0
- **Description:** Fetch Day-Ahead and Real-Time prices from CAISO OASIS API
- **Requirements:**
  - Support configurable date ranges (handle multi-month requests)
  - **Automatic chunking**: Split requests >30 days into 30-day chunks (CAISO API limit)
  - Sequential fetching of chunks with progress tracking
  - Concatenate chunked results into single dataset
  - Handle rate limiting with exponential backoff
  - Retry failed requests (configurable max retries)
  - Cache downloaded data to avoid redundant API calls
  - Support skip-if-exists option for individual chunks
  - Validate data completeness (check for missing timestamps)
  - Detect and handle gaps between chunks
- **Acceptance Criteria:**
  - Successfully fetch 30-day price data within 2 minutes
  - Successfully fetch 90-day price data (3 chunks) within 6 minutes
  - Handle rate limiting without failures
  - No duplicate timestamps at chunk boundaries
  - Detect and report incomplete data
  - Save raw data in standardized CSV format
  - Log chunk progress (e.g., "Fetching chunk 2/3: 2025-08-01 to 2025-08-31")

**FR-4: Weather Data Fetching**
- **Priority:** P0
- **Description:** Fetch hourly weather data from Open-Meteo API
- **Requirements:**
  - Support configurable weather variables (42 variables)
  - Support arbitrary date ranges (Open-Meteo has no hard limit but may batch for performance)
  - Optional chunking for very large requests (>90 days) to improve reliability
  - Handle API rate limits (10,000 requests/day)
  - Retry on network failures
  - Validate geographic coordinates
  - Check for missing or anomalous weather values
  - Concatenate results if chunked
- **Acceptance Criteria:**
  - Fetch 30-day hourly weather data within 1 minute
  - Fetch 90-day hourly weather data within 3 minutes
  - Handle all 42 weather variables correctly
  - Detect gaps in weather data
  - Save in standardized format with proper timestamps

**FR-5: Data Existence Checking**
- **Priority:** P1
- **Description:** Check if data already exists before fetching
- **Requirements:**
  - Compare existing data date ranges with requested ranges
  - Support `skip_if_exists` flag in config
  - Verify data integrity before skipping fetch
  - Log when using cached vs fresh data
- **Acceptance Criteria:**
  - Correctly identify when data exists
  - Validate data integrity (file size, record count)
  - Clear logging of data source (cached vs API)

### 3.3 Data Preprocessing

**FR-6: Weather Data Interpolation**
- **Priority:** P0
- **Description:** Interpolate hourly weather to 5-minute intervals
- **Requirements:**
  - Support multiple interpolation methods (cubic, linear, circular, forward-fill)
  - Auto-select appropriate method per variable type
  - Handle circular data (wind direction) correctly
  - Configurable target frequency (5min default)
  - Validate interpolated values are within reasonable ranges
- **Acceptance Criteria:**
  - Interpolate 720 hourly records to 8,640 5-min records
  - Wind direction handles 0°/360° boundary correctly
  - No NaN values in interpolated output
  - Execution time <30 seconds for 30-day dataset

**FR-7: Dataset Merging**
- **Priority:** P0
- **Description:** Merge weather and price data on aligned timestamps
- **Requirements:**
  - Inner join on timestamp (only keep matching records)
  - Handle timezone differences
  - Log number of merged vs dropped records
  - Validate no duplicate timestamps
  - Save merged dataset with clear naming
- **Acceptance Criteria:**
  - Successfully merge 8,640 records
  - Zero duplicate timestamps
  - All timestamps are 5-minute intervals
  - Missing data report generated

**FR-8: Feature Engineering**
- **Priority:** P0
- **Description:** Create derived features for model training
- **Requirements:**
  - **Temporal features:** hour, day_of_week, is_weekend, is_peak_hours
  - **Price lags:** Configurable lag steps (default: 1-24)
  - **Rolling statistics:** Mean, std, min, max over configurable windows
  - **Weather features:** Temperature conversions, wind speed conversions
  - **Derived indicators:** price_negative, price_high, price volatility
  - Support custom feature definitions via config
  - Save feature metadata (names, types, transformations)
- **Acceptance Criteria:**
  - Generate 111+ features as specified in config
  - Handle edge cases at start of timeseries (NaN for lags)
  - Feature metadata JSON saved with model
  - Execution time <1 minute

### 3.4 Data Splitting

**FR-9: Train/Validation/Test Split**
- **Priority:** P0
- **Description:** Split data chronologically into train/val/test sets
- **Requirements:**
  - Respect temporal ordering (no future leakage)
  - Support configurable split ratios (default: 80/10/10)
  - Maintain minimum window size for rolling features
  - Save split indices or separate CSV files
  - Log date ranges for each split
- **Acceptance Criteria:**
  - Splits sum to 100% of data
  - No temporal overlap between sets
  - Test set is most recent data
  - Clear documentation of split boundaries

### 3.5 Model Training

**FR-10: XGBoost Model Training**
- **Priority:** P0
- **Description:** Train XGBoost regression model for price forecasting
- **Requirements:**
  - Support all XGBoost hyperparameters via config
  - Implement n-step ahead forecasting (configurable horizon)
  - Use lookback window for features (configurable)
  - Fit scaler on training data only
  - Save model, scaler, and feature columns
  - Track training time and resource usage
- **Acceptance Criteria:**
  - Achieve R² > 0.99 on test set (baseline performance)
  - Training completes in <5 minutes for 8,640 samples
  - Model artifacts saved in versioned directory
  - Feature importance plot generated

**FR-11: Multi-Model Support**
- **Priority:** P2
- **Description:** Support training multiple model types
- **Requirements:**
  - Extensible architecture for new models (LSTM, LightGBM, etc.)
  - Enable/disable models via config
  - Train enabled models in sequence or parallel
  - Save each model's artifacts separately
  - Compare model performance in report
- **Acceptance Criteria:**
  - Add new model type with <100 lines of code
  - Config flag correctly enables/disables models
  - Each model has isolated artifacts directory

**FR-12: Model Evaluation**
- **Priority:** P0
- **Description:** Evaluate trained models on test set
- **Requirements:**
  - Calculate metrics: RMSE, MAE, R², MAPE
  - Generate prediction vs actual plots
  - Create residual analysis plots
  - Calculate feature importance
  - Save all metrics to JSON
  - Log evaluation results
- **Acceptance Criteria:**
  - All metrics calculated correctly
  - Plots saved to `plots/training/` directory
  - Metrics JSON is machine-readable
  - Execution time <30 seconds

### 3.6 Validation

**FR-13: Arbitrage Strategy Validation**
- **Priority:** P1
- **Description:** Run rolling intrinsic arbitrage on test set
- **Requirements:**
  - Use trained model for price forecasting
  - Execute rolling intrinsic optimization
  - Calculate trading metrics (revenue, efficiency, actions)
  - Compare against baseline strategies (naive, persistence)
  - Generate arbitrage visualization (9-panel plot)
  - Save detailed trading log
- **Acceptance Criteria:**
  - Validation runs without errors
  - Trading metrics calculated correctly
  - Visualization matches `visualize_arbitrage_results.py` output
  - Execution time <2 minutes

**FR-14: Strategy Comparison**
- **Priority:** P2
- **Description:** Compare trained model against baseline strategies
- **Requirements:**
  - Implement baseline strategies (persistence, naive, day-ahead only)
  - Run all strategies on same test set
  - Calculate relative performance improvements
  - Generate comparison table
  - Statistical significance testing
- **Acceptance Criteria:**
  - All baselines run successfully
  - Comparison table shows percentage improvements
  - Clear winner identified in report

### 3.7 Reporting

**FR-15: Pipeline Summary Report**
- **Priority:** P1
- **Description:** Generate comprehensive training report
- **Requirements:**
  - **Executive Summary:** Key metrics, model performance, recommendations
  - **Data Quality:** Records fetched, missing values, outliers detected
  - **Model Performance:** All evaluation metrics, plots
  - **Validation Results:** Trading metrics, strategy comparison
  - **Configuration:** Full config file included
  - **Execution Metadata:** Run time, timestamps, versions
  - Support Markdown and HTML formats
  - Include all generated plots inline
- **Acceptance Criteria:**
  - Report generated automatically at pipeline completion
  - All sections populated with correct data
  - Plots render correctly in HTML
  - Report saved to `reports/` directory

**FR-16: Logging & Monitoring**
- **Priority:** P1
- **Description:** Comprehensive logging throughout pipeline
- **Requirements:**
  - Timestamped logs for each pipeline stage
  - Progress indicators for long-running operations
  - Warning logs for data quality issues
  - Error logs with full stack traces
  - Summary statistics logged at each stage
  - Support log levels (DEBUG, INFO, WARNING, ERROR)
  - Save logs to file and optionally print to console
- **Acceptance Criteria:**
  - All major operations logged
  - Log files <10 MB for typical run
  - Errors include actionable troubleshooting info
  - Log rotation for long-running sessions

### 3.8 Error Handling & Recovery

**FR-17: Checkpointing**
- **Priority:** P1
- **Description:** Save pipeline state for resume capability
- **Requirements:**
  - Save checkpoint after each major stage
  - Store completed stages, data paths, intermediate results
  - Support `--resume` flag to restart from last checkpoint
  - Clear checkpoint after successful completion
  - Handle corrupted checkpoints gracefully
- **Acceptance Criteria:**
  - Pipeline resumes from correct stage
  - No data re-fetched unnecessarily
  - Checkpoint file <1 MB
  - Resume works after any stage failure

**FR-18: API Retry Logic**
- **Priority:** P0
- **Description:** Robust retry mechanism for API failures
- **Requirements:**
  - Configurable max retries per API call
  - Exponential backoff between retries
  - Respect rate limits (CAISO, Open-Meteo)
  - Log all retry attempts
  - Fail gracefully after max retries
  - Different retry strategies per API
- **Acceptance Criteria:**
  - Handle transient network errors automatically
  - No unnecessary retries on 4xx errors
  - Rate limit errors trigger appropriate delays
  - All retries logged with reason

**FR-19: Data Validation**
- **Priority:** P1
- **Description:** Validate data quality at each stage
- **Requirements:**
  - Check for missing timestamps
  - Detect outliers in price data (>3 std dev)
  - Validate weather values in reasonable ranges
  - Check for duplicate records
  - Verify expected record counts
  - Generate data quality report
  - Warnings don't stop pipeline; errors do
- **Acceptance Criteria:**
  - All validation checks implemented
  - Clear distinction between warnings and errors
  - Data quality report saved
  - Common issues automatically fixed (e.g., duplicates)

---

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-1: Execution Time**
- Complete 30-day training pipeline in <15 minutes (excluding API calls)
- Data collection: <5 minutes
- Preprocessing: <2 minutes
- Training: <5 minutes
- Validation: <3 minutes

**NFR-2: Resource Usage**
- Peak memory usage <8 GB for 30-day dataset
- CPU usage scales with available cores
- Disk usage <500 MB for all artifacts

**NFR-3: Scalability**
- Support up to 90-day training windows (3x current)
- Handle 10+ models in single pipeline run
- Process 100K+ records without performance degradation

### 4.2 Reliability

**NFR-4: Error Rate**
- Pipeline success rate >95% for normal conditions
- Automatic recovery from transient API failures
- No silent failures (all errors logged and reported)

**NFR-5: Data Integrity**
- Zero data loss during processing
- Checksums for downloaded data
- Atomic writes for model artifacts

### 4.3 Maintainability

**NFR-6: Code Quality**
- 80%+ test coverage for core pipeline logic
- Type hints for all public functions
- Comprehensive docstrings (Google style)
- PEP 8 compliant code
- Maximum function complexity: 10 (cyclomatic)

**NFR-7: Documentation**
- User guide with examples
- API documentation (auto-generated)
- Configuration schema documented
- Common troubleshooting scenarios
- Architecture diagrams

### 4.4 Usability

**NFR-8: User Experience**
- Single command execution with sensible defaults
- Progress bars for long operations
- Clear error messages with suggested fixes
- Configuration examples provided
- Dry-run mode for validation

**NFR-9: Compatibility**
- Python 3.8+ support
- Works on macOS, Linux, Windows
- No system-level dependencies beyond pip packages
- Backward compatible with existing data files

---

## 5. Technical Specifications

### 5.1 Technology Stack

**Languages & Frameworks:**
- Python 3.8+
- PyYAML for config parsing
- Click or argparse for CLI
- Joblib for model serialization

**Libraries:**
- pandas, numpy (data processing)
- xgboost, scikit-learn (ML)
- matplotlib, seaborn (visualization)
- requests (API calls)
- logging (standard library)

**Infrastructure:**
- File-based storage (no database required)
- Local execution (no cloud dependencies)
- Optional: Docker containerization

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     train_pipeline.py                        │
│                     (Main Orchestrator)                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──► PipelineConfig (config validation)
             │
             ├──► DataCollector (CAISO + Weather APIs)
             │    ├──► DateRangeChunker (splits >30 day requests)
             │    ├──► Cache Manager
             │    └──► API Rate Limiter
             │
             ├──► DataPreprocessor
             │    ├──► Interpolator
             │    ├──► Merger
             │    └──► FeatureEngineer
             │
             ├──► DataSplitter (train/val/test)
             │
             ├──► ModelTrainer
             │    ├──► XGBoostTrainer
             │    ├──► LSTMTrainer (future)
             │    └──► ModelEvaluator
             │
             ├──► PipelineValidator
             │    └──► ArbitrageBacktester
             │
             ├──► PipelineReporter
             │    ├──► PlotGenerator
             │    └──► ReportWriter
             │
             └──► CheckpointManager
```

### 5.3 Data Flow

```
Config File
    ↓
[Stage 1: Data Collection]
CAISO API → Raw Price Data → data/raw/caiso_*.csv
Weather API → Raw Weather Data → data/raw/weather_*.csv
    ↓
[Stage 2: Preprocessing]
Interpolation → data/processed/weather_5min.csv
Merging → data/processed/merged_dataset.csv
Feature Engineering → data/processed/features.csv
    ↓
[Stage 3: Splitting]
Train/Val/Test Sets → data/processed/{train,val,test}.csv
    ↓
[Stage 4: Training]
Model Training → models/xgboost_YYYYMMDD/
    ├── model.pkl
    ├── scaler.pkl
    ├── config.json
    └── metrics.json
    ↓
[Stage 5: Validation]
Arbitrage Backtest → results/arbitrage_results.csv
    ↓
[Stage 6: Reporting]
Summary Report → reports/summary_YYYYMMDD.html
```

### 5.4 Configuration Schema

See Section 3.1 for detailed configuration structure. Full JSON schema will be provided in `config/schema.json` for validation.

### 5.5 API Contracts

**Internal Module APIs:**

```python
class DataCollector:
    def fetch_caiso_data(self, start_date: str, end_date: str) -> pd.DataFrame
    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame
    def _chunk_date_range(self, start: str, end: str, max_days: int = 30) -> List[Tuple[str, str]]

class DataPreprocessor:
    def interpolate(self, df: pd.DataFrame, method: str) -> pd.DataFrame
    def merge(self, weather: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame

class ModelTrainer:
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Model
    def evaluate(self, model: Model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict
    def save(self, model: Model, path: str) -> None
```

---

## 6. User Stories

### 6.1 Data Scientist: Quick Experimentation

**As a** data scientist
**I want to** easily experiment with different feature sets
**So that** I can quickly identify the best predictors for price forecasting

**Acceptance Criteria:**
- Modify feature configuration in YAML
- Run pipeline with single command
- Compare results across experiments
- Execution time <10 minutes per experiment

### 6.2 ML Engineer: Production Training

**As an** ML engineer
**I want to** schedule automated model retraining
**So that** models stay current with recent market conditions

**Acceptance Criteria:**
- Pipeline runs via cron/scheduler
- Automatic error notifications
- Model artifacts versioned by date
- Training reports emailed/saved

### 6.3 Researcher: Reproducible Results

**As a** researcher
**I want to** exactly reproduce previous training runs
**So that** I can validate published results and build upon them

**Acceptance Criteria:**
- Config file captures all parameters
- Same config produces identical results
- Random seeds controlled
- Version information captured

### 6.4 Operations: Reliable Deployment

**As an** operations engineer
**I want to** reliable pipeline execution with clear error handling
**So that** production deployments don't fail unexpectedly

**Acceptance Criteria:**
- Pipeline fails fast with clear errors
- Automatic retry for transient issues
- Checkpointing enables resume
- Comprehensive logging for debugging

### 6.5 Data Scientist: Long Historical Analysis

**As a** data scientist
**I want to** train models on 6+ months of historical data
**So that** I can capture seasonal patterns and improve model accuracy

**Acceptance Criteria:**
- Configure 180-day training window in YAML
- Pipeline automatically chunks CAISO requests (6 chunks of 30 days)
- Progress tracking shows "Fetching chunk 3/6..."
- All chunks successfully concatenated
- No manual intervention required
- Execution time scales linearly with date range

---

## 7. Implementation Plan

### 7.1 Phases

**Phase 1: Core Pipeline (Weeks 1-2)**
- Configuration management (FR-1, FR-2)
- Data collection (FR-3, FR-4, FR-5)
- Basic error handling (FR-18)
- CLI interface

**Phase 2: Preprocessing & Features (Week 3)**
- Interpolation (FR-6)
- Merging (FR-7)
- Feature engineering (FR-8)
- Data splitting (FR-9)

**Phase 3: Model Training (Week 4)**
- XGBoost trainer (FR-10)
- Model evaluation (FR-12)
- Multi-model support (FR-11)

**Phase 4: Validation & Reporting (Week 5)**
- Arbitrage validation (FR-13)
- Pipeline reports (FR-15)
- Logging system (FR-16)

**Phase 5: Robustness (Week 6)**
- Checkpointing (FR-17)
- Data validation (FR-19)
- Strategy comparison (FR-14)
- Testing & documentation

### 7.2 Dependencies

- Existing scripts provide reference implementations
- Current model achieves R² = 0.993 (baseline target)
- CAISO and Open-Meteo APIs are stable
- No new infrastructure required

### 7.3 Testing Strategy

**Unit Tests:**
- Each module tested independently
- Mock API responses
- Edge case handling
- 80%+ code coverage

**Integration Tests:**
- End-to-end pipeline with sample data
- Config validation scenarios
- Error recovery paths
- Checkpoint resume

**Performance Tests:**
- Benchmark against baseline (current manual process)
- Memory profiling
- Scalability tests (30, 60, 90 day windows)

**User Acceptance Testing:**
- Data scientists run experiments
- Operations team tests scheduled runs
- Validate against existing model results

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| API rate limit changes | Medium | High | Configurable limits, fallback to cached data |
| Data format changes | Low | High | Schema validation, version pinning |
| Memory issues with large datasets | Medium | Medium | Chunked processing, streaming where possible |
| Configuration complexity | High | Medium | Sensible defaults, validation, examples |
| Backward compatibility breaks | Low | High | Semantic versioning, deprecation warnings |
| Checkpoint corruption | Low | Medium | Checksums, graceful fallback |
| Model performance regression | Medium | High | Baseline metrics, automated alerts |
| Chunk boundary timestamp gaps | Medium | Medium | Validate timestamp continuity, detect/report gaps |
| CAISO 31-day limit enforcement | Low | High | Automatic chunking, warn if chunks needed |

---

## 9. Open Questions

1. **Q:** Should we support distributed training for very large datasets?
   **A:** Defer to Phase 2; current dataset size doesn't require it

2. **Q:** What's the strategy for handling breaking config changes?
   **A:** Use semantic versioning; provide migration scripts

3. **Q:** Should we integrate with MLflow or similar experiment tracking?
   **A:** Nice-to-have for Phase 2; start with simple JSON metrics

4. **Q:** How do we handle timezone issues with CAISO data?
   **A:** Standardize on UTC internally; convert for reporting

5. **Q:** Should preprocessing be cached separately from raw data?
   **A:** Yes - cache interpolated/merged data with hash of input files

6. **Q:** For CAISO chunking, should chunks overlap to avoid edge effects?
   **A:** No overlap needed - CAISO timestamps are continuous. Validate no gaps between chunks and log any missing intervals.

7. **Q:** Should individual chunks be cached separately or only the concatenated result?
   **A:** Cache both - individual chunks for partial retries, concatenated for reuse. Use date-based naming: `caiso_chunk_20250801_20250831.csv`

---

## 10. Success Criteria

The pipeline will be considered successful when:

1. ✅ Single command executes complete training workflow
2. ✅ 95%+ pipeline success rate in testing
3. ✅ <15 minute end-to-end execution time
4. ✅ Config file enables exact reproduction of results
5. ✅ Automated report generated with all key metrics
6. ✅ Error messages are actionable and clear
7. ✅ Documentation enables new users to run pipeline in <30 minutes
8. ✅ Test coverage >80% for core functionality
9. ✅ Checkpoint resume works for all failure scenarios
10. ✅ Model performance matches or exceeds baseline (R² > 0.99)

---

## 11. Future Enhancements

**Post-MVP Features:**
- Hyperparameter optimization (Optuna integration)
- Multi-site training orchestration
- Real-time data quality dashboard
- Automated model deployment to production
- A/B testing framework for model comparison
- Cloud storage backend (S3, GCS)
- Distributed training support (Dask, Ray)
- Interactive configuration UI
- Airflow/Prefect workflow integration
- Continuous training on new data

---

## 12. Appendix

### 12.1 Glossary

- **DA**: Day-Ahead (CAISO market)
- **RT**: Real-Time (CAISO market)
- **LMP**: Locational Marginal Price
- **SOC**: State of Charge
- **Rolling Intrinsic**: Dynamic programming optimization for battery arbitrage
- **CAISO**: California Independent System Operator
- **OASIS**: Open Access Same-time Information System (CAISO's API)

### 12.2 References

- [CAISO OASIS API Documentation](http://oasis.caiso.com/mrioasis/logon.do)
- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Current codebase: `/Users/hari/Desktop/aqi/battery_arbitrage/`

### 12.3 Stakeholders

- **Project Owner**: Battery Arbitrage Team
- **Technical Lead**: ML Engineering Team
- **Users**: Data Scientists, ML Engineers, Researchers
- **Reviewers**: Operations Team, Energy Trading Team

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-04 | Claude | Initial draft |

