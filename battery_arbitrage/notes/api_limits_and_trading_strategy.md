# API Limits and Trading Strategy

## Overview

Real-world battery trading systems must respect API rate limits while maintaining continuous operations. This document explains how to manage API limits intelligently for effective trading.

## API Limits Analysis

### 1. Open-Meteo Weather API
- **Free Tier Limit**: 10,000 requests per day
- **Rate Limit**: ~600 requests per hour
- **No API key required** for basic usage
- **Cache Duration**: 1 hour (3600 seconds)

### 2. CAISO OASIS API
- **Data Range Limit**: Maximum 31 days per request
- **Rate Limit**: ~60 requests per minute
- **HTTP 429**: Rate limiting response (requires backoff)
- **No daily limit** but must respect rate limiting

## Trading Strategy with API Limits

### Conservative Limits (For Production)
To ensure reliable operation, we use conservative limits:
- **Weather API**: 100 calls/day (1% of actual limit)
- **CAISO API**: 10 calls/minute (17% of actual limit)

### API Call Schedule

With 100 weather API calls per day:

```
24 hours / 100 calls = 0.24 hours/call = 14.4 minutes/call
→ Round to 15-minute intervals = 96 calls/day
```

**Optimized Schedule**:
- **Weather updates**: Every 15 minutes (4 per hour)
- **Price updates**: Every 5 minutes with 5-minute cache
- **Trading decisions**: Every 5 minutes (288/day)

## Implementation Strategy

### 1. Intelligent Caching

```python
class DataCache:
    - Weather data: 15-minute cache
    - Price data: 5-minute cache
    - Persistent disk storage for backup
    - Cache hit rate target: >85%
```

### 2. API Call Distribution

**Peak Hours (6 AM - 10 PM)**: 16 hours
- Weather: Every 15 minutes → 64 calls
- Prices: Cached aggressively

**Off-Peak (10 PM - 6 AM)**: 8 hours
- Weather: Every 30 minutes → 16 calls
- Prices: Reduced frequency

**Daily Total**: 80 weather calls (under 100 limit)

### 3. Fallback Mechanisms

When API limits are reached:

1. **Weather Data Fallback**:
   - Use cached data up to 2 hours old
   - Interpolate between known points
   - Use persistence forecast (assume no change)
   - Default to seasonal averages

2. **Price Data Fallback**:
   - Use cached data up to 15 minutes old
   - Apply time-of-day patterns
   - Use day-ahead prices as backup
   - Historical average for time slot

### 4. Trading Decision Flow

```
Every 5 minutes:
1. Check if forecast update needed (every hour)
   - If yes and API available → Fetch new data
   - If API limited → Use cached forecast

2. Get latest prices
   - Check 5-minute cache first
   - If stale and API available → Fetch
   - If API limited → Use interpolation

3. Execute Rolling Intrinsic
   - Use best available forecast
   - Make trading decision
   - Update battery state

4. Log API usage and cache stats
```

## Example Daily Operation

### Morning (6 AM - 12 PM)
- **Weather calls**: 24 (every 15 min)
- **Price calls**: ~72 (with caching)
- **Trading decisions**: 72
- **Cache hit rate**: ~60%

### Afternoon (12 PM - 6 PM)
- **Weather calls**: 24
- **Price calls**: ~50 (improved caching)
- **Trading decisions**: 72
- **Cache hit rate**: ~80%

### Evening (6 PM - 12 AM)
- **Weather calls**: 24
- **Price calls**: ~40
- **Trading decisions**: 72
- **Cache hit rate**: ~90%

### Night (12 AM - 6 AM)
- **Weather calls**: 12 (reduced frequency)
- **Price calls**: ~20
- **Trading decisions**: 72
- **Cache hit rate**: ~95%

**Daily Totals**:
- Weather API: 84 calls (under 100 limit)
- CAISO API: ~182 calls (well distributed)
- Trading decisions: 288
- Average cache hit rate: ~81%

## Benefits of API-Aware Design

### 1. Reliability
- Never exceeds API limits
- Graceful degradation when limits approached
- Continuous operation even with API outages

### 2. Efficiency
- Minimizes redundant API calls
- High cache hit rates reduce latency
- Intelligent scheduling optimizes data freshness

### 3. Cost Optimization
- Stays within free tier limits
- Reduces potential API costs
- Efficient use of computational resources

### 4. Performance
- 288 trading decisions per day
- 5-minute response to market changes
- Forecast updates every hour during peak times

## Configuration Examples

### Aggressive (Maximum API Usage)
```python
APILimits(
    weather_daily_limit=1000,  # 10% of limit
    weather_calls_per_hour=40,  # Every 1.5 minutes
    caiso_calls_per_minute=30,  # 50% of limit
    trading_interval_minutes=1,  # Trade every minute
)
```

### Conservative (Minimal API Usage)
```python
APILimits(
    weather_daily_limit=50,     # 0.5% of limit
    weather_calls_per_hour=2,   # Every 30 minutes
    caiso_calls_per_minute=5,   # 8% of limit
    trading_interval_minutes=15, # Trade every 15 minutes
)
```

### Recommended Production Settings
```python
APILimits(
    weather_daily_limit=100,    # 1% of limit
    weather_calls_per_hour=4,   # Every 15 minutes
    caiso_calls_per_minute=10,  # 17% of limit
    trading_interval_minutes=5,  # Trade every 5 minutes
)
```

## Monitoring and Alerts

### Key Metrics to Track
1. **API Usage**
   - Calls per hour/day
   - Proximity to limits
   - Failed calls

2. **Cache Performance**
   - Hit rate by data type
   - Cache age distribution
   - Memory usage

3. **Trading Performance**
   - Decisions with fresh vs stale data
   - Forecast accuracy degradation
   - Revenue impact of API limits

### Alert Thresholds
- Weather API > 80% of daily limit → Warning
- Weather API > 95% of daily limit → Critical
- Cache hit rate < 70% → Investigate
- Consecutive API failures > 3 → Alert

## Conclusion

By implementing intelligent caching, scheduled updates, and graceful fallbacks, the battery trading system can operate continuously while respecting API limits. The system makes 288 trading decisions daily using only 100 weather API calls, achieving high efficiency through strategic data management.