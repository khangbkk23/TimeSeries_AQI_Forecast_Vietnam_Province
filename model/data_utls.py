import os, shutil
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import json

def create_mappings(info_path):
    with open(info_path, 'r', encoding='utf-8-sig') as f:
        dataset_info = json.load(f)

    station_to_idx = {}
    region_to_idx = {'north': 0, 'middle': 1, 'south': 2}
    station_region_lookup = {}

    for entry in dataset_info.values():
        file_name = entry.get('file_name')
        region_str = entry.get('region')
        
        if not file_name:
            continue
        station_id_str = '_'.join(file_name.split('_')[:-1])
        
        if station_id_str not in station_to_idx:
            s_idx = len(station_to_idx)
            station_to_idx[station_id_str] = s_idx
            station_region_lookup[s_idx] = region_to_idx.get(region_str, 0)
            
    return station_to_idx, region_to_idx, station_region_lookup

class AQIDualEmbeddingDataset(Dataset):
    def __init__(self, data_dir, station_map, station_region_lookup, sequence_length=14, target_col='VN_AQI'):
        self.sequence_length = sequence_length
        self.samples = []
        
        file_paths = glob.glob(os.path.join(data_dir, "*_processed.csv"))
        
        if not file_paths:
            print(f" Không tìm thấy file csv nào trong {data_dir}")

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            station_id_str = filename.replace('_processed.csv', '')
            
            if station_id_str not in station_map:
                continue 
                
            s_idx = station_map[station_id_str]
            r_idx = station_region_lookup[s_idx]
            
            df = pd.read_csv(file_path)
            
            exclude_cols = ['Date', 'station_id', 'region', target_col]
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            features = df[feature_cols].values.astype(np.float32)
            targets = df[target_col].values.astype(np.float32)
            
            num_records = len(df)
            if num_records > sequence_length:
                for i in range(num_records - sequence_length):
                    seq_data = features[i : i + sequence_length]
                    target_val = targets[i + sequence_length]
                    
                    self.samples.append({
                        'sequence': seq_data,
                        'station_idx': s_idx,
                        'region_idx': r_idx,
                        'target': target_val
                    })
        
        if len(self.samples) > 0:
            self.input_dim = self.samples[0]['sequence'].shape[1]
        else:
            self.input_dim = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['sequence']),
            torch.tensor(sample['station_idx'], dtype=torch.long),
            torch.tensor(sample['region_idx'], dtype=torch.long),
            torch.tensor(sample['target'])
        )
        
def get_dataloaders(config):
    s_map, r_map, s_r_lookup = create_mappings(config['info_path'])
    
    print(f"Đang tải dữ liệu train từ {config['train_dir']}...")
    train_ds = AQIDualEmbeddingDataset(config['train_dir'], s_map, s_r_lookup, config['sequence_length'])
    
    print(f"Đang tải dữ liệu validation từ {config['val_dir']}...")
    val_ds = AQIDualEmbeddingDataset(config['val_dir'], s_map, s_r_lookup, config['sequence_length'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    info = {
        'num_stations': len(s_map),
        'num_regions': len(r_map),
        'input_dim': train_ds.input_dim
    }
    
    return train_loader, val_loader, info