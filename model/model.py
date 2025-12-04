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
        
        # 3. Attention layer
        self.attention = Attention(config.hidden_dim)
        
        # 3. Regressor
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_station, x_region):
        s_vec = self.station_emb(x_station) # [Batch, S_Dim]
        r_vec = self.region_emb(x_region)   # [Batch, R_Dim]
        
        combined_vec = torch.cat([s_vec, r_vec], dim=1)
        seq_len = x_seq.size(1)
        emb_seq = combined_vec.unsqueeze(1).repeat(1, seq_len, 1)
        
        final_input = torch.cat([x_seq, emb_seq], dim=2)
        
        lstm_out, _ = self.lstm(final_input)
        
        context_vector, _ = self.attention(lstm_out)
        
        return self.regressor(context_vector).squeeze()