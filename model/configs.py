import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    info_path = './data/origin/dataset_info.json'
    train_dir = './data/train/'
    val_dir = './data/validation/'
    test_dir = './data/test/'
    save_model_path = 'best_aqi_model.pth'

    sequence_length = 21
    embedding_dim_station = 8
    embedding_dim_region = 4
    hidden_dim = 64
    num_layers = 2
    dropout = 0.3
    
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 15              
    
    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = ModelConfig()