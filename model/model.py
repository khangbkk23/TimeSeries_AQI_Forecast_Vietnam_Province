import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):

        # Calculate Energy score
        attn_weights = F.softmax(self.attn(lstm_output), dim=1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        
        return context_vector, attn_weights
    
class DualEmbeddingLSTM(nn.Module):
    def __init__(self, config, num_stations, num_regions, input_dim):
        """
        Args:
            config: Object chứa các tham số
            num_stations: Tổng số trạm
            num_regions: Tổng số vùng
            input_dim: Số lượng feature đầu vào từ dữ liệu
        """
        super(DualEmbeddingLSTM, self).__init__()
        
        # 1. Embedding layers
        self.station_emb = nn.Embedding(num_stations, config.embedding_dim_station)
        self.region_emb = nn.Embedding(num_regions, config.embedding_dim_region)
        
        # 2. LSTM layer
        lstm_input_size = input_dim + config.embedding_dim_station + config.embedding_dim_region
        
        lstm_input_size = input_dim + config.embedding_dim_station + config.embedding_dim_region
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        fusion_dim = config.hidden_dim * 2
        
        # 3. Attention layer
        
        self.attention = Attention(config.hidden_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_station, x_region):
        s_vec = self.station_emb(x_station) 
        r_vec = self.region_emb(x_region)   
        combined_vec = torch.cat([s_vec, r_vec], dim=1)
        seq_len = x_seq.size(1)
        emb_seq = combined_vec.unsqueeze(1).repeat(1, seq_len, 1)
        final_input = torch.cat([x_seq, emb_seq], dim=2)
        lstm_out, _ = self.lstm(final_input) 
        lstm_out = self.layer_norm(lstm_out)
        context_vector, attn_weights = self.attention(lstm_out)
        last_hidden = lstm_out[:, -1, :] 
        combined_features = torch.cat([context_vector, last_hidden], dim=1) 
        return self.regressor(combined_features).squeeze()
    
class WeightedMSELoss(nn.Module):
    def __init__(self, high_val_weight=2.0, threshold=1.0):
        """
        Args:
            high_val_weight: Hệ số phạt
            threshold: Ngưỡng xác định đỉnh.
        """
        super().__init__()
        self.high_val_weight = high_val_weight
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        
        high_val_mask = target > self.threshold
        weights = torch.ones_like(loss)
        weights[high_val_mask] = self.high_val_weight
        
        return torch.mean(loss * weights)