import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self, high_val_weight=2.0, threshold=1.0):
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
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim, 1)

    def forward(self, lstm_output):
        energy = self.attn(lstm_output) 
        attn_weights = F.softmax(energy, dim=1) 
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector, attn_weights

# [MỚI] Class Model đã được nâng cấp
class TripleEmbeddingBiLSTM(nn.Module):
    def __init__(self, config, num_stations, num_regions, num_provinces, input_dim):
        super(TripleEmbeddingBiLSTM, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1
        
        # 1. Embedding layers
        self.station_emb = nn.Embedding(num_stations, config.embedding_dim_station)
        self.region_emb = nn.Embedding(num_regions, config.embedding_dim_region)
        self.province_emb = nn.Embedding(num_provinces, config.embedding_dim_province)
        
        # 2. LSTM layer
        # Input size = Features + StationEmb + RegionEmb + ProvinceEmb
        lstm_input_size = (input_dim + 
                           config.embedding_dim_station + 
                           config.embedding_dim_region + 
                           config.embedding_dim_province)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        lstm_output_dim = config.hidden_dim * self.num_directions
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # 3. Attention layer
        self.attention = Attention(lstm_output_dim)
        
        # 4. Deep Regressor
        fusion_dim = lstm_output_dim * 2 
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_station, x_region, x_province):
        s_vec = self.station_emb(x_station) 
        r_vec = self.region_emb(x_region)
        p_vec = self.province_emb(x_province) # [MỚI]
        
        # Fusion: Nối 3 vector lại [Batch, Emb_S + Emb_R + Emb_P]
        combined_vec = torch.cat([s_vec, r_vec, p_vec], dim=1) 
        
        # Repeat vector này cho toàn bộ sequence length
        seq_len = x_seq.size(1)
        emb_seq = combined_vec.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate với Time-series input
        final_input = torch.cat([x_seq, emb_seq], dim=2)
        
        # LSTM Processing
        lstm_out, (hidden, cell) = self.lstm(final_input)
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention
        context_vector, attn_weights = self.attention(lstm_out)

        # Extract Last Hidden State
        if self.bidirectional:
            hidden_reshaped = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            last_forward = hidden_reshaped[-1, 0, :, :] 
            last_backward = hidden_reshaped[-1, 1, :, :]
            last_hidden_state = torch.cat((last_forward, last_backward), dim=1)
        else:
            last_hidden_state = hidden[-1, :, :]

        combined_features = torch.cat([context_vector, last_hidden_state], dim=1)
        
        return self.regressor(combined_features).squeeze()