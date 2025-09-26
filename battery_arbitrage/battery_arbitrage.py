"""
Minimal Battery Arbitrage Implementation
Based on the rolling intrinsic strategy from the paper
Adapted for CAISO 5-minute real-time market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BatteryConfig:
    """Battery system configuration parameters"""
    capacity_kwh: float = 500.0  # Battery capacity in kWh
    max_power_kw: float = 100.0  # Max charge/discharge power in kW
    efficiency_charge: float = 0.95  # Charging efficiency
    efficiency_discharge: float = 0.95  # Discharging efficiency
    degradation_cost_per_kwh: float = 0.004  # Degradation cost in $/kWh
    trading_cost_per_kwh: float = 0.00009  # Trading cost in $/kWh
    min_soc_kwh: float = 0.0  # Minimum state of charge
    max_soc_kwh: float = 500.0  # Maximum state of charge


class RollingIntrinsic:
    """
    Implements a simplified rolling intrinsic policy for battery arbitrage
    using dynamic programming approach
    """

    def __init__(self, config: BatteryConfig):
        self.config = config
        self.total_cost = config.degradation_cost_per_kwh + config.trading_cost_per_kwh

        # DP grid parameters
        self.soc_grid_points = 51  # Number of discretization points for SoC
        self.soc_grid = np.linspace(
            config.min_soc_kwh,
            config.max_soc_kwh,
            self.soc_grid_points
        )

        # Time interval for 5-minute data (in hours)
        self.dt = 5 / 60  # 5 minutes = 1/12 hour

    def calculate_profit(self, action_kw: float, price: float) -> float:
        """
        Calculate profit for a given action and price
        Positive action = charging (buying), negative = discharging (selling)
        """
        if action_kw > 0:  # Charging (buying power)
            # Cost of buying power including efficiency losses
            energy_kwh = action_kw * self.dt
            cost = (price + self.total_cost) * energy_kwh
            return -cost
        elif action_kw < 0:  # Discharging (selling power)
            # Revenue from selling power accounting for efficiency
            energy_kwh = -action_kw * self.dt
            revenue = (price - self.total_cost) * energy_kwh
            return revenue
        return 0.0

    def get_feasible_actions(self, soc: float) -> List[float]:
        """
        Get feasible charge/discharge actions given current SoC
        Returns list of power actions in kW
        """
        actions = []

        # Maximum energy that can be charged in one interval
        max_charge_energy = min(
            self.config.max_power_kw * self.dt,
            (self.config.max_soc_kwh - soc) / self.config.efficiency_charge
        )

        # Maximum energy that can be discharged in one interval
        max_discharge_energy = min(
            self.config.max_power_kw * self.dt,
            soc * self.config.efficiency_discharge
        )

        # Create action grid (simplified - using 11 actions)
        num_actions = 11

        # Charging actions (positive)
        if max_charge_energy > 0:
            charge_actions = np.linspace(0, max_charge_energy / self.dt, num_actions // 2 + 1)
            actions.extend(charge_actions[1:])  # Exclude 0

        # Discharging actions (negative)
        if max_discharge_energy > 0:
            discharge_actions = np.linspace(0, -max_discharge_energy / self.dt, num_actions // 2 + 1)
            actions.extend(discharge_actions[1:])  # Exclude 0

        # Always include no action
        actions.append(0.0)

        return actions

    def update_soc(self, soc: float, action_kw: float) -> float:
        """Update state of charge based on action"""
        if action_kw > 0:  # Charging
            energy_added = action_kw * self.dt * self.config.efficiency_charge
            new_soc = min(soc + energy_added, self.config.max_soc_kwh)
        elif action_kw < 0:  # Discharging
            energy_removed = -action_kw * self.dt / self.config.efficiency_discharge
            new_soc = max(soc - energy_removed, self.config.min_soc_kwh)
        else:
            new_soc = soc

        return new_soc

    def solve_dp(self, prices: List[float], initial_soc: float,
                 horizon: Optional[int] = None) -> Tuple[List[float], List[float]]:
        """
        Solve the rolling intrinsic optimization using dynamic programming

        Args:
            prices: List of prices for future time steps ($/kWh)
            initial_soc: Initial state of charge (kWh)
            horizon: Planning horizon (number of time steps), if None use all prices

        Returns:
            actions: List of optimal actions (kW) for each time step
            soc_trajectory: List of SoC values after each action
        """
        if horizon is None:
            horizon = len(prices)
        else:
            horizon = min(horizon, len(prices))

        # Initialize value function
        T = horizon
        value_function = np.zeros((T + 1, self.soc_grid_points))
        policy = np.zeros((T, self.soc_grid_points))

        # Backward pass - compute value functions
        for t in range(T - 1, -1, -1):
            price = prices[t]

            for i, soc in enumerate(self.soc_grid):
                best_value = -np.inf
                best_action = 0.0

                # Evaluate all feasible actions
                for action in self.get_feasible_actions(soc):
                    # Calculate immediate profit
                    immediate_profit = self.calculate_profit(action, price)

                    # Calculate next state
                    next_soc = self.update_soc(soc, action)

                    # Interpolate future value
                    future_value = np.interp(next_soc, self.soc_grid, value_function[t + 1])

                    # Total value
                    total_value = immediate_profit + future_value

                    if total_value > best_value:
                        best_value = total_value
                        best_action = action

                value_function[t, i] = best_value
                policy[t, i] = best_action

        # Forward pass - execute policy
        actions = []
        soc_trajectory = [initial_soc]
        current_soc = initial_soc

        for t in range(T):
            # Interpolate optimal action for current SoC
            optimal_action = np.interp(current_soc, self.soc_grid, policy[t])

            # Ensure action is feasible
            feasible_actions = self.get_feasible_actions(current_soc)
            if len(feasible_actions) > 0:
                # Find closest feasible action
                optimal_action = min(feasible_actions,
                                    key=lambda x: abs(x - optimal_action))
            else:
                optimal_action = 0.0

            actions.append(optimal_action)

            # Update SoC
            current_soc = self.update_soc(current_soc, optimal_action)
            soc_trajectory.append(current_soc)

        return actions, soc_trajectory


class BatteryArbitrageBacktest:
    """Simple backtesting framework for battery arbitrage strategies"""

    def __init__(self, strategy: RollingIntrinsic):
        self.strategy = strategy
        self.results = None

    def load_caiso_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess CAISO data"""
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df

    def run_backtest(self, data: pd.DataFrame,
                     rolling_window: int = 12,  # 1 hour window (12 * 5min)
                     reoptimize_freq: int = 1) -> pd.DataFrame:
        """
        Run rolling intrinsic backtest on historical data

        Args:
            data: DataFrame with CAISO price data
            rolling_window: Planning horizon in number of time steps
            reoptimize_freq: How often to reoptimize (in time steps)
        """
        results = []
        current_soc = self.strategy.config.capacity_kwh / 2  # Start at 50% SoC

        for i in range(0, len(data) - rolling_window, reoptimize_freq):
            # Get price forecast for next window
            price_window = data['rt_price_kwh'].iloc[i:i+rolling_window].values

            # Solve optimization
            actions, soc_trajectory = self.strategy.solve_dp(
                price_window,
                current_soc,
                rolling_window
            )

            # Execute actions for reoptimize_freq steps
            for j in range(min(reoptimize_freq, len(actions))):
                if i + j >= len(data):
                    break

                action = actions[j]
                price = data['rt_price_kwh'].iloc[i + j]

                # Calculate revenue/cost
                if action > 0:  # Charging
                    energy_kwh = action * self.strategy.dt
                    cost = price * energy_kwh * (1 + self.strategy.total_cost)
                    revenue = -cost
                elif action < 0:  # Discharging
                    energy_kwh = -action * self.strategy.dt
                    revenue = price * energy_kwh * (1 - self.strategy.total_cost)
                else:
                    revenue = 0

                # Update SoC
                new_soc = self.strategy.update_soc(current_soc, action)

                results.append({
                    'timestamp': data.index[i + j],
                    'price': price,
                    'action_kw': action,
                    'energy_kwh': action * self.strategy.dt,
                    'soc_before': current_soc,
                    'soc_after': new_soc,
                    'revenue': revenue,
                    'cumulative_revenue': 0  # Will be calculated later
                })

                current_soc = new_soc

        # Convert to DataFrame and calculate cumulative revenue
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df['cumulative_revenue'] = results_df['revenue'].cumsum()

        self.results = results_df
        return results_df

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if self.results is None or self.results.empty:
            return {}

        metrics = {
            'total_revenue': self.results['revenue'].sum(),
            'avg_daily_revenue': self.results.groupby(
                self.results['timestamp'].dt.date
            )['revenue'].sum().mean(),
            'total_energy_charged': self.results[
                self.results['energy_kwh'] > 0
            ]['energy_kwh'].sum(),
            'total_energy_discharged': -self.results[
                self.results['energy_kwh'] < 0
            ]['energy_kwh'].sum(),
            'avg_soc': self.results['soc_after'].mean(),
            'num_cycles': (
                self.results['energy_kwh'].abs().sum() /
                (2 * self.strategy.config.capacity_kwh)
            )
        }

        return metrics


def main():
    """Example usage of the battery arbitrage system"""

    # Initialize battery configuration
    battery_config = BatteryConfig(
        capacity_kwh=500.0,
        max_power_kw=100.0,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        degradation_cost_per_kwh=0.004,
        trading_cost_per_kwh=0.00009
    )

    # Create strategy
    strategy = RollingIntrinsic(battery_config)

    # Create backtester
    backtester = BatteryArbitrageBacktest(strategy)

    # Load data
    print("Loading CAISO data...")
    data = backtester.load_caiso_data('casio_data.txt')
    print(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")

    # Run backtest with 1-hour rolling window, reoptimizing every 5 minutes
    print("\nRunning backtest...")
    results = backtester.run_backtest(
        data,
        rolling_window=12,  # 1 hour = 12 * 5min intervals
        reoptimize_freq=1   # Reoptimize every 5 minutes
    )

    # Calculate and display metrics
    metrics = backtester.calculate_metrics()

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Revenue: ${metrics.get('total_revenue', 0):.2f}")
    print(f"Average Daily Revenue: ${metrics.get('avg_daily_revenue', 0):.2f}")
    print(f"Total Energy Charged: {metrics.get('total_energy_charged', 0):.2f} kWh")
    print(f"Total Energy Discharged: {metrics.get('total_energy_discharged', 0):.2f} kWh")
    print(f"Average SoC: {metrics.get('avg_soc', 0):.2f} kWh")
    print(f"Number of Cycles: {metrics.get('num_cycles', 0):.2f}")

    # Save results
    if not results.empty:
        results.to_csv('arbitrage_results.csv', index=False)
        print(f"\nResults saved to arbitrage_results.csv")

        # Display sample of results
        print("\nSample of trading decisions (first 10 non-zero actions):")
        action_results = results[results['action_kw'] != 0].head(10)
        for _, row in action_results.iterrows():
            action_type = "Charge" if row['action_kw'] > 0 else "Discharge"
            print(f"{row['timestamp']}: {action_type} {abs(row['action_kw']):.2f} kW @ ${row['price']*1000:.2f}/MWh -> Revenue: ${row['revenue']:.4f}")


if __name__ == "__main__":
    main()