# Rolling Intrinsic Algorithm - How It Works

## Overview

The Rolling Intrinsic (RI) algorithm is a sophisticated battery arbitrage strategy that makes continuous trading decisions by solving dynamic programming optimization problems. Unlike simple "buy low, sell high" strategies, it considers both immediate profits and future opportunities.

## Core Concept

### What Makes It "Intrinsic"
- **Myopic Decision Making**: Focuses on immediate profitability without speculation
- **No Future Price Speculation**: Uses current market information only
- **Risk Averse**: Only enters positions that are immediately profitable

### What Makes It "Rolling"
- **Continuous Reoptimization**: Re-solves the optimization every 5 minutes
- **Sliding Window**: Always looks ahead 1 hour (12 × 5-minute intervals)
- **Adaptive**: Adjusts strategy as new price information arrives

## State and Decision Framework

### Primary State Variable
**State of Charge (SoC)**: Current battery energy level (0-500 kWh)

### Possible Actions
1. **Charge** (+kW): Buy electricity, store energy (pay cost)
2. **Discharge** (-kW): Sell electricity, release energy (earn revenue)
3. **Hold** (0 kW): No action, maintain current state

### Physical Constraints
```python
# Power limits
max_charge_power = 100 kW
max_discharge_power = 100 kW

# Energy limits
min_soc = 0 kWh
max_soc = 500 kWh

# Efficiency losses
charge_efficiency = 95%
discharge_efficiency = 95%
round_trip_efficiency = 90%
```

## How Decisions Are Made

### 1. Price Information Source

**Current Implementation (Backtesting)**:
```python
# Perfect foresight over rolling window
price_window = data['rt_price_kwh'].iloc[i:i+12].values
# Gets actual next 12 prices (1 hour of 5-min intervals)
```

**Real-World Implementation**:
- Order book data (bid/ask stacks)
- Price forecasting models
- Market signals and indicators

### 2. Dynamic Programming Process

#### Backward Pass (Policy Creation)

Starting from the **last time period** and working backwards:

```python
# For each time period (from T-1 down to 0)
for t in range(T-1, -1, -1):
    price_t = prices[t]

    # For each possible battery state
    for soc in [0, 10, 20, ..., 500]:  # kWh
        best_value = -infinity
        best_action = 0

        # Evaluate all feasible actions
        for action in feasible_actions(soc):  # kW

            # Calculate immediate profit/loss
            immediate_profit = calculate_profit(action, price_t)

            # Calculate resulting battery state
            next_soc = update_soc(soc, action)

            # Look up future value from next time period
            future_value = value_function[t+1][next_soc]

            # Total value = immediate + future
            total_value = immediate_profit + future_value

            # Keep track of best option
            if total_value > best_value:
                best_value = total_value
                best_action = action

        # Store optimal policy
        policy[t][soc] = best_action
        value_function[t][soc] = best_value
```

#### Forward Pass (Execution)

Execute the pre-computed policy:

```python
current_soc = 250  # Current battery state
current_time = 0

# Look up optimal action for current state
optimal_action = policy[current_time][current_soc]

# Execute the action
revenue = execute_trade(optimal_action, current_price)
new_soc = update_soc(current_soc, optimal_action)
```

## Detailed Example Walkthrough

### Setup
- **Current SoC**: 250 kWh (50% full)
- **Future Prices**: [20, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25] $/MWh
- **Current Time**: Period 0 (price = $20/MWh)

### Backward Pass Calculation

**Period 11** (Final period, price = $25):
```
At SoC = 250 kWh:
- No future periods to consider
- Best action: Do nothing (value = 0)
- Value[11][250] = 0
```

**Period 10** (price = $30):
```
At SoC = 250 kWh:
Option 1 - Discharge 100kW:
  - Energy: 100kW × (5/60)h = 8.33 kWh
  - Revenue: $30/MWh × 8.33 kWh × 0.95 = $2.37
  - New SoC: 250 - 8.33/0.95 = 241.2 kWh
  - Future Value: Value[11][241.2] = 0
  - Total Value: $2.37 + $0 = $2.37 ✓ BEST

Option 2 - Hold:
  - Immediate: $0
  - Future: Value[11][250] = 0
  - Total: $0

Option 3 - Charge 100kW:
  - Cost: $30/MWh × 8.33 kWh = -$2.50
  - Future: Value[11][257.9] = 0
  - Total: -$2.50

Decision: Discharge (Value[10][250] = $2.37)
```

**Period 9** (price = $35):
```
At SoC = 250 kWh:
Option 1 - Discharge:
  - Immediate: $35 × 8.33 × 0.95 = $2.76
  - Future: Value[10][241.2] = $2.37
  - Total: $2.76 + $2.37 = $5.13 ✓ BEST

Option 2 - Hold:
  - Immediate: $0
  - Future: Value[10][250] = $2.37
  - Total: $2.37

Decision: Discharge (Value[9][250] = $5.13)
```

**Continuing backwards through all periods...**

**Period 0** (Current time, price = $20):
```
At SoC = 250 kWh:
The algorithm discovers that despite the immediate cost of charging at $20,
the future value of having more energy when prices hit $50 makes it profitable.

Decision: CHARGE (because Value[1][257.9] > immediate cost)
```

### Forward Pass Execution

```python
# Time 0: Current state SoC = 250 kWh, Price = $20/MWh
action = policy[0][250]  # Returns: Charge 100kW
revenue = -$20 × 8.33 = -$1.67  # Pay to charge
new_soc = 250 + (8.33 × 0.95) = 257.9 kWh

# Time 1: SoC = 257.9 kWh, Price = $25/MWh
action = policy[1][257.9]  # Returns: Hold or small charge
# ... continue for all periods
```

## Decision Rules and Logic

### When to Charge (Buy Power)
- **Current price LOW** + **Future prices HIGH**
- **Battery has capacity** remaining
- **Future value** of stored energy > **immediate cost**

### When to Discharge (Sell Power)
- **Current price HIGH** + **Future prices LOWER**
- **Battery has energy** to discharge
- **Immediate revenue** > **future value** of stored energy

### When to Hold
- **Current price MODERATE** + **Better opportunities ahead**
- **At capacity limits** (can't charge when full, can't discharge when empty)
- **Future uncertainty** makes waiting optimal

## Key Insights

### 1. Not Simple Buy-Low-Sell-High
- Makes **91 decisions per day** (every 5 minutes)
- Continuously **rebalances** position
- Can charge **multiple times** during low price periods
- Can discharge **partially** during price spikes

### 2. Physical Constraints Drive Strategy
- **Round-trip efficiency losses** (10%) must be overcome
- **Power limits** prevent instant arbitrage
- **Capacity limits** force strategic timing

### 3. Perfect Information Advantage
- Uses **actual future prices** in backtest
- Real-world performance depends on **forecast accuracy**
- Algorithm quality separate from prediction quality

### 4. Optimization vs Execution
- **Backward pass**: Computationally intensive planning
- **Forward pass**: Fast lookup and execution
- **Reoptimization**: Adapts to new information

## Performance Results

From our implementation:
- **Total Revenue**: $55.88 over 47 hours
- **Efficiency**: 95.8% round-trip
- **Activity**: 80.9% active trading (charging/discharging)
- **Strategy**: 44.4% charging, 36.5% discharging, 19.1% holding

## Real-World Considerations

### Forecast Quality
```python
# Backtest uses perfect foresight
price_window = actual_future_prices[i:i+12]

# Real trading needs forecasts
price_window = forecast_model.predict(current_time, horizon=12)
```

### Market Impact
- Algorithm assumes **price-taking** behavior
- Large batteries might **influence prices**
- **Execution delays** affect profitability

### Risk Management
- **Forecast errors** can cause losses
- **Market conditions** change over time
- **Degradation costs** accumulate with cycling

## Algorithm Strengths

1. **Systematic Approach**: No emotional decisions
2. **Rapid Adaptation**: Responds to price changes
3. **Optimal Under Constraints**: Maximizes value given physical limits
4. **Scalable**: Works for different battery sizes
5. **Transparent**: Clear decision logic

## Algorithm Limitations

1. **Requires Accurate Forecasts**: Garbage in, garbage out
2. **Myopic**: No long-term strategic planning
3. **Computationally Intensive**: DP solving every 5 minutes
4. **Price-Taking**: Assumes no market impact
5. **Perfect Execution**: Real trading has delays and slippage

## Conclusion

The Rolling Intrinsic algorithm transforms battery arbitrage from simple rules-based trading into a sophisticated optimization problem that balances immediate profits with future opportunities, all while respecting the physical constraints of energy storage systems.