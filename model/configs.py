import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    info_path = './data/origin/dataset_info.json'
    train_dir = './data/train/'
    val_dir = './data/validation/'
    test_dir = './data/test/'
    save_model_path = 'best_aqi_model.pth'

    
    scheduler_factor = 0.5
    scheduler_patience = 6
    min_lr = 1e-7
    warmup_epochs = 3
    warmup_start_lr = 1e-5
    grad_clip = 1.0
    # loss_type = 'weighted_mse'
    loss_type = 'huber'
    threshold = 0.5 
    high_val_weight = 2.5
    
    sequence_length = 24
    embedding_dim_station = 10
    embedding_dim_region = 4
    
    hidden_dim = 64
    num_layers = 2
    bidirectional = True
    
    dropout = 0.3
    
    batch_size = 32
    learning_rate = 0.0005
    epochs = 100
    patience = 25        
    
    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = ModelConfig()