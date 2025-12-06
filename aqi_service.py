import torch
import joblib
import json
import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Import class model và config
from model.model import DualEmbeddingBiLSTM
from model.configs import cfg

class AQIPredictor:
    def __init__(self, model_path='./ai_engine/weights/best_aqi_model_dual.pth'):
        print("Đang khởi tạo AQI Predictor...")
        self.device = torch.device('cpu')
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        self.station_enc = joblib.load(os.path.join(base_path, 'encoders/station_encoder.pkl'))
        self.region_enc = joblib.load(os.path.join(base_path, 'encoders/region_encoder.pkl'))
        self.scaler = joblib.load(os.path.join(base_path, 'normalized_data/global_scaler.pkl'))
        
        json_path = os.path.join(base_path, 'data/origin/dataset_info.json')
        self.province_map = self._build_province_mapping(json_path)
        
        # Khởi tạo Model
        num_stations = len(self.station_enc.classes_)
        num_regions = len(self.region_enc.classes_)
        input_dim = 24 
        
        self.model = DualEmbeddingBiLSTM(
            config=cfg,
            num_stations=num_stations,
            num_regions=num_regions,
            input_dim=input_dim
        ).to(self.device)
        
        full_model_path = os.path.join(base_path, '../', model_path)
        if os.path.exists(model_path):
             checkpoint = torch.load(model_path, map_location=self.device)
        else:
             checkpoint = torch.load(os.path.join(base_path, 'weights/best_aqi_model_dual.pth'), map_location=self.device)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def _build_province_mapping(self, json_path):
        if not os.path.exists(json_path):
            print(f"Không tìm thấy {json_path}")
            return {}

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mapping = {}
        for key, info in data.items():
            prov = info.get('province')
            region = info.get('region')
            filename = info.get('file_name')
            
            if prov and filename:
                station_id = '_'.join(filename.split('_')[:-1])
                if prov not in mapping:
                    mapping[prov] = {
                        'region': region, 
                        'default_station': station_id
                    }
        return mapping

    def _add_engineered_features(self, df):
        df = df.copy()
        
        # 1. Temporal Features
        df['Date'] = pd.to_datetime(df['Date'])
        df['hour'] = df['Date'].dt.hour
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 2. Lag Features (Lags = 1, 7)
        for col in ['VN_AQI', 'PM-2-5', 'PM-10']:
            if col in df.columns:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_7'] = df[col].shift(7)
        
        # 3. Rolling features (Window = 7)
        for col in ['VN_AQI', 'PM-2-5']:
            if col in df.columns:
                df[f'{col}_roll_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
        
        df = df.bfill().ffill()
        return df

    def preprocess_sequence(self, df_history):
        # 1. Feature Engineering
        df_processed = self._add_engineered_features(df_history)
        
        # 2. Chọn cột Feature
        feature_cols = [
            'VN_AQI', 'CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2', 
            'hour', 'hour_sin', 'hour_cos', 
            'day_of_week', 'month', 'quarter', 
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'VN_AQI_lag_1', 'VN_AQI_lag_7', 
            'PM-2-5_lag_1', 'PM-2-5_lag_7', 
            'PM-10_lag_1', 'PM-10_lag_7', 
            'VN_AQI_roll_mean_7', 'PM-2-5_roll_mean_7'
        ]
        try:
            data_window = df_processed[feature_cols].tail(24)
        except KeyError as e:
            print(f"Lỗi thiếu cột dữ liệu: {e}")
            return None

        # 3. Scale dữ liệu
        try:
            scaled_data = self.scaler.transform(data_window)
        except Exception as e:
            print(f"Lỗi Scaling: {e}")
            return None
        return torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _update_history_for_recursive(self, current_df, predicted_aqi):
        last_row = current_df.iloc[-1].copy()
        next_date = last_row['Date'] + timedelta(days=1)
        
        new_row = last_row.copy()
        new_row['Date'] = next_date
        new_row['VN_AQI'] = predicted_aqi
        
        return pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    def predict_next_7_days(self, df_history, province_name):
        if province_name not in self.province_map:
            print(f"Không tìm thấy thông tin cho tỉnh: {province_name}")
            return []
            
        info = self.province_map[province_name]
        station_id = info['default_station']
        region = info['region']
        
        try:
            s_idx = torch.tensor([self.station_enc.transform([station_id])[0]], dtype=torch.long).to(self.device)
            r_idx = torch.tensor([self.region_enc.transform([region])[0]], dtype=torch.long).to(self.device)
        except Exception as e:
            print(f"❌ Lỗi Encoding trạm/vùng: {e}")
            return []

        future_predictions = []
        running_history = df_history.copy()
        running_history['Date'] = pd.to_datetime(running_history['Date'])

        print(f"Bắt đầu dự báo 7 ngày cho: {province_name} (Trạm: {station_id})")

        for i in range(7):
            # Biến đổi dữ liệu -> Tensor
            input_tensor = self.preprocess_sequence(running_history)
            
            if input_tensor is None:
                break

            # Dự báo
            with torch.no_grad():
                pred_val = self.model(input_tensor, s_idx, r_idx).item()
            
            # Lưu kết quả
            last_date = running_history.iloc[-1]['Date']
            next_date = last_date + timedelta(days=1)
            
            pred_val = max(0, min(pred_val, 500))
            
            future_predictions.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "aqi": round(pred_val, 2)
            })
            running_history = self._update_history_for_recursive(running_history, pred_val)

        return future_predictions