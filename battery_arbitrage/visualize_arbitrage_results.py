#!/usr/bin/env python3
"""
Battery Arbitrage Results Visualization

This script creates comprehensive visualizations for battery arbitrage results
from the rolling intrinsic strategy analysis.

Author: Generated for Eland Solar & Storage Center analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import numpy as np

def load_arbitrage_data(filepath):
    """Load and preprocess arbitrage results data."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date

    # Categorize actions
    df['action_type'] = 'Hold'
    df.loc[df['action_kw'] > 0, 'action_type'] = 'Discharge'
    df.loc[df['action_kw'] < 0, 'action_type'] = 'Charge'

    # Calculate daily metrics
    daily_stats = df.groupby('date').agg({
        'revenue': 'sum',
        'cumulative_revenue': 'last',
        'action_kw': lambda x: (x != 0).sum(),  # Number of actions
        'price': ['min', 'max', 'mean']
    }).round(4)

    # Flatten column names
    daily_stats.columns = ['revenue', 'cumulative_revenue', 'action_kw', 'price_min', 'price_max', 'price_mean']

    return df, daily_stats

def create_comprehensive_plots(df, daily_stats):
    """Create comprehensive battery arbitrage visualization."""

    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Plot 1: State of Charge Over Time
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(df['timestamp'], df['soc_after'], linewidth=2, color='steelblue', alpha=0.8)
    plt.title('Battery State of Charge Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('State of Charge (kWh)')
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.grid(True, alpha=0.3)

    # Plot 2: Actions vs Electricity Prices
    ax2 = plt.subplot(3, 3, 2)
    charge_mask = df['action_kw'] < 0
    discharge_mask = df['action_kw'] > 0
    hold_mask = df['action_kw'] == 0

    plt.scatter(df.loc[charge_mask, 'timestamp'], df.loc[charge_mask, 'price'],
               c='red', alpha=0.6, s=30, label='Charge')
    plt.scatter(df.loc[discharge_mask, 'timestamp'], df.loc[discharge_mask, 'price'],
               c='green', alpha=0.6, s=30, label='Discharge')
    plt.scatter(df.loc[hold_mask, 'timestamp'], df.loc[hold_mask, 'price'],
               c='gray', alpha=0.3, s=10, label='Hold')

    plt.title('Battery Actions vs Electricity Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price ($/MWh)')
    plt.legend()
    plt.xticks(rotation=45)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.grid(True, alpha=0.3)

    # Plot 3: Cumulative Revenue Progression
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(df['timestamp'], df['cumulative_revenue'], linewidth=3, color='darkgreen')
    plt.title('Cumulative Revenue Progression', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Revenue ($)')
    plt.xticks(rotation=45)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.grid(True, alpha=0.3)

    # Add final revenue annotation
    final_revenue = df['cumulative_revenue'].iloc[-1]
    plt.annotate(f'Final Revenue: ${final_revenue:.2f}',
                xy=(df['timestamp'].iloc[-1], final_revenue),
                xytext=(0.7, 0.8), textcoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 4: Daily Revenue Breakdown
    ax4 = plt.subplot(3, 3, 4)
    daily_revenue = daily_stats['revenue'].values
    dates = [str(date) for date in daily_stats.index]
    bars = plt.bar(dates, daily_revenue, color='lightblue', edgecolor='navy', linewidth=1.5)
    plt.title('Daily Revenue Breakdown', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Revenue ($)')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, daily_revenue):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'${value:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Plot 5: Action Distribution
    ax5 = plt.subplot(3, 3, 5)
    action_counts = df['action_type'].value_counts()
    colors = ['lightcoral', 'lightgreen', 'lightgray']
    wedges, texts, autotexts = plt.pie(action_counts.values, labels=action_counts.index,
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Distribution of Battery Actions', fontsize=14, fontweight='bold')

    # Plot 6: Price Distribution with Actions
    ax6 = plt.subplot(3, 3, 6)
    plt.hist(df.loc[charge_mask, 'price'], bins=20, alpha=0.7, label='Charge Prices', color='red')
    plt.hist(df.loc[discharge_mask, 'price'], bins=20, alpha=0.7, label='Discharge Prices', color='green')
    plt.title('Price Distribution for Battery Actions', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($/MWh)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: Hourly Action Pattern
    ax7 = plt.subplot(3, 3, 7)
    hourly_actions = df.groupby(['hour', 'action_type']).size().unstack(fill_value=0)
    hourly_actions.plot(kind='bar', stacked=True, ax=ax7, color=['lightcoral', 'lightgreen', 'lightgray'])
    plt.title('Hourly Battery Action Patterns', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Actions')
    plt.xticks(rotation=0)
    plt.legend(title='Action Type')
    plt.grid(True, alpha=0.3, axis='y')

    # Plot 8: Revenue vs SOC Correlation
    ax8 = plt.subplot(3, 3, 8)
    plt.scatter(df['soc_after'], df['revenue'], alpha=0.6, c=df['price'], cmap='viridis')
    plt.colorbar(label='Price ($/MWh)')
    plt.title('Revenue vs State of Charge', fontsize=14, fontweight='bold')
    plt.xlabel('State of Charge (kWh)')
    plt.ylabel('Revenue ($)')
    plt.grid(True, alpha=0.3)

    # Plot 9: Battery Efficiency Metrics
    ax9 = plt.subplot(3, 3, 9)

    # Calculate efficiency metrics
    total_energy_charged = df.loc[df['action_kw'] < 0, 'energy_kwh'].sum() * -1  # Make positive
    total_energy_discharged = df.loc[df['action_kw'] > 0, 'energy_kwh'].sum()
    total_revenue = df['cumulative_revenue'].iloc[-1]

    # Round-trip efficiency
    efficiency = (total_energy_discharged / total_energy_charged * 100) if total_energy_charged > 0 else 0

    # Revenue per kWh
    revenue_per_kwh = total_revenue / total_energy_discharged if total_energy_discharged > 0 else 0

    metrics = ['Total Revenue\n($)', 'Energy Charged\n(kWh)', 'Energy Discharged\n(kWh)',
               'Round-trip\nEfficiency (%)', 'Revenue per\nkWh ($/kWh)']
    values = [total_revenue, total_energy_charged, total_energy_discharged, efficiency, revenue_per_kwh]

    bars = plt.bar(range(len(metrics)), values, color=['gold', 'lightcoral', 'lightgreen', 'lightblue', 'orange'])
    plt.title('Battery Performance Metrics', fontsize=14, fontweight='bold')
    plt.xticks(range(len(metrics)), metrics, rotation=45, ha='right')
    plt.ylabel('Value')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        if i == 3:  # Efficiency percentage
            label = f'{value:.1f}%'
        elif i in [0, 4]:  # Money values
            label = f'${value:.2f}'
        else:  # Energy values
            label = f'{value:.1f}'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.grid(True, alpha=0.3, axis='y')

    # Adjust layout and save
    plt.tight_layout(pad=3.0)

    return fig, {
        'total_revenue': total_revenue,
        'total_energy_charged': total_energy_charged,
        'total_energy_discharged': total_energy_discharged,
        'efficiency': efficiency,
        'revenue_per_kwh': revenue_per_kwh,
        'daily_stats': daily_stats
    }

def print_summary_statistics(df, metrics):
    """Print summary statistics of the arbitrage results."""
    print("\n" + "="*80)
    print("BATTERY ARBITRAGE ANALYSIS SUMMARY")
    print("="*80)

    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   • Total Revenue: ${metrics['total_revenue']:.2f}")
    print(f"   • Energy Charged: {metrics['total_energy_charged']:.1f} kWh")
    print(f"   • Energy Discharged: {metrics['total_energy_discharged']:.1f} kWh")
    print(f"   • Round-trip Efficiency: {metrics['efficiency']:.1f}%")
    print(f"   • Revenue per kWh: ${metrics['revenue_per_kwh']:.3f}/kWh")

    print(f"\n🔋 BATTERY OPERATIONS:")
    action_counts = df['action_type'].value_counts()
    total_actions = len(df)
    print(f"   • Total Time Points: {total_actions}")
    print(f"   • Charge Actions: {action_counts.get('Charge', 0)} ({action_counts.get('Charge', 0)/total_actions*100:.1f}%)")
    print(f"   • Discharge Actions: {action_counts.get('Discharge', 0)} ({action_counts.get('Discharge', 0)/total_actions*100:.1f}%)")
    print(f"   • Hold Actions: {action_counts.get('Hold', 0)} ({action_counts.get('Hold', 0)/total_actions*100:.1f}%)")

    print(f"\n💰 PRICE ANALYSIS:")
    print(f"   • Min Price: ${df['price'].min():.4f}/MWh")
    print(f"   • Max Price: ${df['price'].max():.4f}/MWh")
    print(f"   • Average Price: ${df['price'].mean():.4f}/MWh")
    print(f"   • Price Volatility (Std): ${df['price'].std():.4f}/MWh")

    print(f"\n📅 DAILY BREAKDOWN:")
    for date, stats in metrics['daily_stats'].iterrows():
        print(f"   • {date}: Revenue ${stats['revenue']:.2f}, "
              f"Actions {stats['action_kw']}, "
              f"Avg Price ${stats['price_mean']:.4f}/MWh")

    print(f"\n🕐 TIME PERIOD:")
    print(f"   • Start: {df['timestamp'].min()}")
    print(f"   • End: {df['timestamp'].max()}")
    print(f"   • Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} hours")

    print("="*80)

def main():
    """Main execution function."""
    print("Loading battery arbitrage results...")

    # Load data
    df, daily_stats = load_arbitrage_data('arbitrage_results.csv')

    print(f"Loaded {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Create visualizations
    print("Creating comprehensive visualizations...")
    fig, metrics = create_comprehensive_plots(df, daily_stats)

    # Save the plot
    output_file = 'battery_arbitrage_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved comprehensive analysis plot to: {output_file}")

    # Print summary statistics
    print_summary_statistics(df, metrics)

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()