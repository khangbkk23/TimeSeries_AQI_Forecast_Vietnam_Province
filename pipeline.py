from math import comb
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler

def load_and_concat_data(info_file_path, data_directory):
    with open(info_file_path, 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)

    all_dfs = []
    for key, info in dataset_info.items():
        file_name = info.get('file_name')
        region = info.get('region')
        if not file_name:
            continue
            
        file_path = os.path.join(data_directory, file_name)
        df = pd.read_csv(file_path, sep=',')
        df['region'] = region
        station_id = '_'.join(file_name.split('_')[:-1]) 
        df['station_id'] = station_id
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True)
    combined_df = combined_df.sort_values(by=['station_id', 'Date'])
    
    return combined_df

def clean_data(df):
    numeric_cols = ['VN_AQI', 'CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_temporal_features(df):
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def add_lag_features(df, lags=[1, 7]):
    df = df.sort_values(by=['station_id', 'Date'])
    for col in ['VN_AQI', 'PM-2-5', 'PM-10']:
        if col in df.columns and df[col].notna().sum() > 0:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('station_id')[col].shift(lag)
    return df

def add_rolling_features(df, windows=[7]):
    df = df.sort_values(by=['station_id', 'Date'])
    for col in ['VN_AQI', 'PM-2-5']:
        if col in df.columns and df[col].notna().sum() > 0:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = (
                    df.groupby('station_id')[col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
    return df

def train_val_test_split_temporal(df, val_cutoff_date_str, test_cutoff_date_str):
    val_cutoff = pd.to_datetime(val_cutoff_date_str)
    test_cutoff = pd.to_datetime(test_cutoff_date_str)
    
    train_df = df[df['Date'] < val_cutoff].copy()
    val_df = df[(df['Date'] >= val_cutoff) & (df['Date'] < test_cutoff)].copy()
    test_df = df[df['Date'] >= test_cutoff].copy()
    
    return train_df, val_df, test_df

def impute_data(train_df, val_df, test_df, features_to_impute):
    imputation_values = {}
    
    for col in features_to_impute:
        train_df[col] = train_df.groupby('station_id')[col].fillna(method='ffill')
        train_df[col] = train_df.groupby('station_id')[col].fillna(method='bfill')
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        imputation_values[col] = median_val
        
        val_df[col] = val_df.groupby('station_id')[col].fillna(method='ffill')
        val_df[col] = val_df[col].fillna(median_val)
        
        test_df[col] = test_df.groupby('station_id')[col].fillna(method='ffill')
        test_df[col] = test_df[col].fillna(median_val)

    return train_df, val_df, test_df, imputation_values

def normalize_data(train_df, val_df, test_df, features_to_scale):
    scaler = StandardScaler()
    scaler.fit(train_df[features_to_scale])
    
    train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
    val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
    
    return train_df, val_df, test_df, scaler

def full_preprocessing_pipeline(combined_df, val_cutoff, test_cutoff, save_paths):
    
    # Clean data
    combined_df = clean_data(combined_df)
    # Add temporal features
    combined_df = add_temporal_features(combined_df)
    
    # Add lag and rolling features
    combined_df = add_lag_features(combined_df, lags=[1, 7])
    combined_df = add_rolling_features(combined_df, windows=[7])
    
    # Split
    train_df, val_df, test_df = train_val_test_split_temporal(
        combined_df, val_cutoff, test_cutoff
    )
    
    # Impute
    pollutant_features = ['CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2', 'VN_AQI']
    lag_features = [col for col in train_df.columns if '_lag_' in col or '_roll_' in col]
    features_to_impute = pollutant_features + lag_features
    
    train_df, val_df, test_df, _ = impute_data(
        train_df, val_df, test_df, features_to_impute
    )
    
    # Normalize
    temporal_cols = ['day_of_week', 'month', 'quarter', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    features_to_scale = [col for col in features_to_impute if col not in temporal_cols]
    
    train_df, val_df, test_df, global_scaler = normalize_data(
        train_df, val_df, test_df, features_to_scale
    )
    
    # Save
    all_stations = combined_df['station_id'].unique()
    data_splits = {
        'train': (train_df, save_paths['train_dir']),
        'validation': (val_df, save_paths['val_dir']),
        'test': (test_df, save_paths['test_dir'])
    }

    for split_name, (data_df, save_dir) in data_splits.items():
        os.makedirs(save_dir, exist_ok=True)
        for station in all_stations:
            file_name = f"{station}_processed.csv"
            station_data = data_df[data_df['station_id'] == station]
            if not station_data.empty:
                station_data.to_csv(os.path.join(save_dir, file_name), sep=',', index=False)
    
    os.makedirs(os.path.dirname(save_paths['scaler_path']), exist_ok=True)
    joblib.dump(global_scaler, save_paths['scaler_path'])

# Run
info_file_path = "./data/origin/dataset_info.json"
data_directory = "./data/origin/"

save_paths = {
    'train_dir': './data/train/',
    'val_dir': './data/validation/',
    'test_dir': './data/test/',
    'scaler_path': './normalized_data/global_scaler.pkl'
}

VAL_CUTOFF_DATE = '2025-04-01'
TEST_CUTOFF_DATE = '2025-07-01'

all_data_df = load_and_concat_data(info_file_path, data_directory)
if all_data_df is not None:
    full_preprocessing_pipeline(all_data_df, VAL_CUTOFF_DATE, TEST_CUTOFF_DATE, save_paths)