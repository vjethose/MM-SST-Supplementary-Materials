#!/usr/bin/env python3
"""
Synthetic dataset generation for MM-SST energy forecasting study.
Generates SIM-D (urban district, 50 buildings, daily) and SIM-B (building zones, 8 zones, hourly).

Usage:
  python sim_data.py --output-dir ./data
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==============================================================================
# SIM-D: Urban District Energy Demand (UDED)
# ==============================================================================

def generate_sim_d(n_buildings=50, n_days=1825, output_dir='./data'):
    """
    Generate SIM-D: 50-building urban district, daily demand, 5 years (1,825 days).

    Demand model:
      Y_i(d) = baseline + trend + seasonality + autocorrelation + weather effect + noise
    """
    np.random.seed(RANDOM_SEED)

    # Initialize arrays
    consumption = np.zeros((n_days, n_buildings))
    temperature = np.zeros(n_days)
    humidity = np.zeros(n_days)
    solar_irradiance = np.zeros(n_days)

    # Generate baseline demand per building (lognormal distribution)
    baseline_log_mean = np.log(50)  # ~50 MWh/day average
    baseline_log_std = 0.4
    baselines = np.exp(np.random.normal(baseline_log_mean, baseline_log_std, n_buildings))

    # Generate spatial locations for k-NN graph
    spatial_locations = np.random.uniform(0, 100, (n_buildings, 2))

    # Generate weather variables
    day_of_year = np.arange(n_days) % 365
    temperature = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, n_days)
    humidity = 60 + 20 * np.cos(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, n_days)
    solar_irradiance = 200 + 150 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 30, n_days)

    # Generate demand for each building
    for i in range(n_buildings):
        # Trend component
        trend = 0.01 * np.arange(n_days) + 0.00001 * np.arange(n_days)**2

        # Seasonality (annual cycle)
        seasonality = 0.2 * baselines[i] * np.sin(2 * np.pi * day_of_year / 365)

        # Autocorrelation (7-day lag)
        autocorr_coef = 0.6
        autocorr = np.zeros(n_days)
        autocorr[7:] = autocorr_coef * consumption[:-7, i]

        # Weather effect
        weather_effect = 0.3 * baselines[i] * (
            0.5 * (20 - temperature) / 10 +  # Temperature sensitivity
            0.1 * (humidity - 60) / 20 +      # Humidity effect
            0.2 * solar_irradiance / 200      # Solar irradiance effect
        )

        # Random shocks (occasional spikes)
        shocks = np.zeros(n_days)
        shock_days = np.random.choice(n_days, size=int(0.02 * n_days), replace=False)
        shocks[shock_days] = np.random.normal(0, 0.1 * baselines[i], len(shock_days))

        # Gaussian noise
        noise = np.random.normal(0, 0.05 * baselines[i], n_days)

        # Combine all components
        consumption[:, i] = (
            baselines[i] + trend + seasonality + autocorr + weather_effect + shocks + noise
        )
        consumption[:, i] = np.maximum(consumption[:, i], 0)  # Ensure non-negative

    # Create DataFrame
    date_range = pd.date_range(start='2018-01-01', periods=n_days, freq='D')
    data = {
        **{f'Building_{i}': consumption[:, i] for i in range(n_buildings)},
        'Date': date_range,
        'Temperature': temperature,
        'Humidity': humidity,
        'SolarIrradiance': solar_irradiance,
        'Month': date_range.month,
        'DayOfWeek': date_range.dayofweek,
    }

    df = pd.DataFrame(data)
    df = df[['Date', 'Month', 'DayOfWeek', 'Temperature', 'Humidity', 'SolarIrradiance'] +
            [f'Building_{i}' for i in range(n_buildings)]]

    # Save dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'sim_d_dataset.csv', index=False)

    # Save spatial locations
    spatial_df = pd.DataFrame(spatial_locations, columns=['X', 'Y'])
    spatial_df['BuildingID'] = [f'Building_{i}' for i in range(n_buildings)]
    spatial_df.to_csv(output_dir / 'sim_d_spatial.csv', index=False)

    # Generate k-NN adjacency graph
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(spatial_locations, metric='euclidean'))

    adjacency = np.zeros((n_buildings, n_buildings))
    for i in range(n_buildings):
        # Find k=5 nearest neighbors
        nearest = np.argsort(distances[i])[1:6]  # Exclude self
        adjacency[i, nearest] = 1
        adjacency[nearest, i] = 1  # Make symmetric

    # Save as JSON
    edges = []
    for i in range(n_buildings):
        for j in np.where(adjacency[i] > 0)[0]:
            if i < j:  # Avoid duplicates
                edges.append([f'Building_{i}', f'Building_{j}'])

    graph_data = {
        'nodes': [f'Building_{i}' for i in range(n_buildings)],
        'edges': edges,
        'adjacency_matrix': adjacency.tolist(),
        'graph_type': 'k-NN (k=5)',
    }

    with open(output_dir / 'sim_d_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)

    # Generate metadata
    metadata = {
        'dataset': 'SIM-D (Urban District Energy Demand)',
        'n_buildings': n_buildings,
        'n_days': n_days,
        'temporal_resolution': 'daily',
        'forecast_horizon': 7,
        'lookback_window': 30,
        'train_test_split': '70/15/15',
        'train_days': int(0.7 * n_days),
        'val_days': int(0.15 * n_days),
        'test_days': int(0.15 * n_days),
        'features': {
            'consumption': n_buildings,
            'weather': 3,
            'temporal': 2,
        },
        'total_features': n_buildings + 5,
        'random_seed': RANDOM_SEED,
    }

    with open(output_dir / 'sim_d_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Generated SIM-D: {n_buildings} buildings, {n_days} days")
    print(f"  - Saved: sim_d_dataset.csv ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  - Graph: sim_d_graph.json ({len(edges)} edges)")

    return df, graph_data

# ==============================================================================
# SIM-B: Building Zone Energy Consumption (SBEC)
# ==============================================================================

def generate_sim_b(n_zones=8, n_hours=2400, output_dir='./data'):
    """
    Generate SIM-B: 8-zone commercial building, hourly demand, 100 days (2,400 hours).

    Demand model:
      Y_i(t) = baseline + hour_effect + day_effect + autocorr + weather + occupancy + noise
    """
    np.random.seed(RANDOM_SEED)

    # Initialize arrays
    consumption = np.zeros((n_hours, n_zones))
    temperature = np.zeros(n_hours)
    humidity = np.zeros(n_hours)
    solar_irradiance = np.zeros(n_hours)
    occupancy = np.zeros((n_hours, n_zones))

    # Generate baseline demand per zone (lognormal)
    baseline_log_mean = np.log(7)  # ~7 kWh/hour average
    baseline_log_std = 0.3
    baselines = np.exp(np.random.normal(baseline_log_mean, baseline_log_std, n_zones))

    # Generate grid positions for zones (2x4 layout)
    zone_positions = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0, 2], [1, 2], [0, 3], [1, 3]
    ], dtype=float)

    # Generate weather variables with hourly cycles
    day_of_year = np.arange(n_hours) // 24 % 365
    hour_of_day = np.arange(n_hours) % 24

    temperature = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + \
                  5 * np.sin(2 * np.pi * hour_of_day / 24) + \
                  np.random.normal(0, 1, n_hours)

    humidity = 60 + 15 * np.cos(2 * np.pi * day_of_year / 365) + \
               10 * np.cos(2 * np.pi * hour_of_day / 24) + \
               np.random.normal(0, 3, n_hours)

    solar_irradiance = np.maximum(200 * np.sin(2 * np.pi * hour_of_day / 24), 0) + \
                       100 * np.sin(2 * np.pi * day_of_year / 365) + \
                       np.random.normal(0, 20, n_hours)

    # Generate occupancy proxy (WiFi signal, ~15% missing)
    for i in range(n_zones):
        occupancy_base = 0.5 * np.sin(2 * np.pi * hour_of_day / 24)  # Peak at midday
        occupancy_base[occupancy_base < 0] = 0
        occupancy[:, i] = occupancy_base + np.random.normal(0, 0.1, n_hours)
        occupancy[:, i] = np.clip(occupancy[:, i], 0, 1)

        # Add 15% missing values
        missing_idx = np.random.choice(n_hours, size=int(0.15 * n_hours), replace=False)
        occupancy[missing_idx, i] = np.nan

    # Generate demand for each zone
    for i in range(n_zones):
        # Hour-of-day effect
        hour_effect = 2 * baselines[i] * np.sin(2 * np.pi * hour_of_day / 24)

        # Day-of-week effect
        day_of_week = np.arange(n_hours) // 24 % 7
        dow_effect = 0.3 * baselines[i] * np.sin(2 * np.pi * day_of_week / 7)

        # 24-hour autocorrelation
        autocorr_coef = 0.5
        autocorr = np.zeros(n_hours)
        autocorr[24:] = autocorr_coef * consumption[:-24, i]

        # Weather effect (temperature and solar irradiance)
        weather_effect = 0.25 * baselines[i] * (
            0.4 * (20 - temperature) / 10 +  # Temperature sensitivity
            0.2 * solar_irradiance / 200      # Solar effect
        )

        # Occupancy effect (WiFi signal strength)
        occupancy_effect = 1.5 * baselines[i] * np.nanmean(occupancy, axis=1)
        occupancy_effect = np.nan_to_num(occupancy_effect, nan=0)

        # Random noise
        noise = np.random.normal(0, 0.1 * baselines[i], n_hours)

        # Combine all components
        consumption[:, i] = (
            baselines[i] + hour_effect + dow_effect + autocorr +
            weather_effect + occupancy_effect + noise
        )
        consumption[:, i] = np.maximum(consumption[:, i], 0)

    # Create DataFrame
    date_range = pd.date_range(start='2023-01-01', periods=n_hours, freq='H')
    data = {
        **{f'Zone_{i}': consumption[:, i] for i in range(n_zones)},
        'Date': date_range,
        'Hour': hour_of_day,
        'DayOfWeek': np.arange(n_hours) // 24 % 7,
        'Temperature': temperature,
        'Humidity': humidity,
        'SolarIrradiance': solar_irradiance,
    }

    # Add occupancy
    for i in range(n_zones):
        data[f'Occupancy_Zone_{i}'] = occupancy[:, i]

    df = pd.DataFrame(data)
    col_order = ['Date', 'Hour', 'DayOfWeek', 'Temperature', 'Humidity', 'SolarIrradiance']
    col_order += [f'Zone_{i}' for i in range(n_zones)]
    col_order += [f'Occupancy_Zone_{i}' for i in range(n_zones)]
    df = df[col_order]

    # Save dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'sim_b_dataset.csv', index=False)

    # Generate grid adjacency graph (2x4 grid)
    edges = [
        ['Zone_0', 'Zone_1'], ['Zone_0', 'Zone_2'],
        ['Zone_1', 'Zone_3'],
        ['Zone_2', 'Zone_3'], ['Zone_2', 'Zone_4'],
        ['Zone_3', 'Zone_5'],
        ['Zone_4', 'Zone_5'], ['Zone_4', 'Zone_6'],
        ['Zone_5', 'Zone_7'],
        ['Zone_6', 'Zone_7'],
    ]

    adjacency = np.zeros((n_zones, n_zones))
    for edge in edges:
        i = int(edge[0].split('_')[1])
        j = int(edge[1].split('_')[1])
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    graph_data = {
        'nodes': [f'Zone_{i}' for i in range(n_zones)],
        'edges': edges,
        'adjacency_matrix': adjacency.tolist(),
        'graph_type': 'Grid (2x4 layout)',
    }

    with open(output_dir / 'sim_b_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)

    # Generate metadata
    metadata = {
        'dataset': 'SIM-B (Single Building Zone Energy Consumption)',
        'n_zones': n_zones,
        'n_hours': n_hours,
        'temporal_resolution': 'hourly',
        'forecast_horizon': 24,
        'lookback_window': 168,
        'train_test_split': '70/15/15',
        'train_hours': int(0.7 * n_hours),
        'val_hours': int(0.15 * n_hours),
        'test_hours': int(0.15 * n_hours),
        'features': {
            'consumption': n_zones,
            'weather': 3,
            'temporal': 2,
            'occupancy': n_zones,
        },
        'total_features': n_zones * 2 + 5,
        'random_seed': RANDOM_SEED,
    }

    with open(output_dir / 'sim_b_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Generated SIM-B: {n_zones} zones, {n_hours} hours")
    print(f"  - Saved: sim_b_dataset.csv ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"  - Graph: sim_b_graph.json ({len(edges)} edges)")

    return df, graph_data

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic datasets for MM-SST')
    parser.add_argument('--output-dir', default='./data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Generate both datasets
    print("Generating synthetic datasets...\n")
    df_d, graph_d = generate_sim_d(output_dir=args.output_dir)
    print()
    df_b, graph_b = generate_sim_b(output_dir=args.output_dir)
    print("\n✅ Done! All datasets saved.")
