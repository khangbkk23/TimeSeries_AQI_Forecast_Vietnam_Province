import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    info_path: str = './data/origin/dataset_info.json'
    train_dir: str = './data/train/'
    val_dir: str = './data/validation/'
    save_model_path: str = 'best_aqi_model.pth'

    sequence_length: int = 14
    embedding_dim_station: int = 8
    embedding_dim_region: int = 4
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 15              
    
    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = ModelConfig()