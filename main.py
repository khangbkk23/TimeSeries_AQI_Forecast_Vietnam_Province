import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from tqdm import tqdm
from model.data_utils import get_dataloaders
from model.model import DualEmbeddingLSTM, WeightedMSELoss
from model.configs import cfg
from model.visualize import plot_learning_curves, plot_prediction_comparison

def warmup_lr_scheduler(epoch, warmup_epochs, warmup_start_lr, base_lr):
    if epoch < warmup_epochs:
        return warmup_start_lr + (base_lr - warmup_start_lr) * epoch / warmup_epochs
    return base_lr

def print_final_report(best_loss, best_epoch, total_epochs):
    print(f"\n{'Summary':^40}")
    print("="*40)
    print(f"Tổng số Epochs        : {total_epochs}")
    print(f"Epoch tốt nhất        : {best_epoch}")
    print(f"Loss tốt nhất         : {best_loss:.6f}")
    print("="*40 + "\n")

def save_results_to_pkl(train_hist, val_hist, best_epoch, best_loss, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_history.pkl")
    
    config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else {}
    data = {
        "train_loss_history": train_hist,
        "val_loss_history": val_hist,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "config": {k: v for k, v in config_dict.items() if not k.startswith('__')}
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Đã xuất file kết quả tại: {save_path}")

def evaluate_test_set(model, test_loader, device):
    model.eval()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    total_mse = 0
    total_mae = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for seqs, s_idxs, r_idxs, targets in test_loader:
            seqs, s_idxs, r_idxs, targets = seqs.to(device), s_idxs.to(device), r_idxs.to(device), targets.to(device)
            
            preds = model(seqs, s_idxs, r_idxs)
            batch_mse = criterion_mse(preds, targets)
            batch_mae = criterion_mae(preds, targets)
            
            total_mse += batch_mse.item() * seqs.size(0)
            total_mae += batch_mae.item() * seqs.size(0)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    num_samples = len(test_loader.dataset)
    final_mse = total_mse / num_samples
    final_mae = total_mae / num_samples
    final_rmse = np.sqrt(final_mse)
    
    print(f"\nFinal test metrics:")
    print(f"   MSE:     {final_mse:.4f}")
    print(f"   RMSE:    {final_rmse:.4f}")
    print(f"   MAE:     {final_mae:.4f}")
    print("-" * 45)
    
    return actuals, predictions

def run_training():
    warmup_epochs = getattr(cfg, 'warmup_epochs', 5)
    warmup_start_lr = getattr(cfg, 'warmup_start_lr', 1e-5)
    grad_clip = getattr(cfg, 'grad_clip', 1.0)
    weight_decay = getattr(cfg, 'weight_decay', 1e-3)
    
    print(f"Device: {cfg.device}")
    print(f"Config: Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.learning_rate}")
    print(f"Adv: Warmup={warmup_epochs}, Clip={grad_clip}, Optim=AdamW")

    config_dict = {
        'info_path': cfg.info_path, 'train_dir': cfg.train_dir,
        'val_dir': cfg.val_dir, 'test_dir': cfg.test_dir,
        'sequence_length': cfg.sequence_length, 'batch_size': cfg.batch_size
    }
    
    train_loader, val_loader, test_loader, data_info = get_dataloaders(config_dict)
    
    if data_info['input_dim'] == 0:
        print("Lỗi: Không tìm thấy dữ liệu.")
        return None, None, [], [], 0, 0, [], []

    model = DualEmbeddingLSTM(
        config=cfg,
        num_stations=data_info['num_stations'],
        num_regions=data_info['num_regions'],
        input_dim=data_info['input_dim']
    ).to(cfg.device)
    
    # Dùng Weighted MSE với threshold thấp và weight vừa phải như đã bàn
    criterion = WeightedMSELoss(high_val_weight=2.5, threshold=0.5)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(cfg.epochs):

        if epoch < warmup_epochs:
            lr = warmup_lr_scheduler(epoch, warmup_epochs, warmup_start_lr, cfg.learning_rate)
            for pg in optimizer.param_groups: pg['lr'] = lr
        
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg.epochs}", leave=False)
        
        for seqs, s_idxs, r_idxs, targets in train_bar:
            seqs, s_idxs, r_idxs, targets = seqs.to(cfg.device), s_idxs.to(cfg.device), r_idxs.to(cfg.device), targets.to(cfg.device)
            
            optimizer.zero_grad()
            predictions = model(seqs, s_idxs, r_idxs)
            loss = criterion(predictions, targets)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            train_losses.append(loss.item())
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = np.mean(train_losses)
        

        model.eval()
        val_weighted_loss_sum = 0
        val_std_mse_sum = 0
        
        with torch.no_grad():
            for seqs, s_idxs, r_idxs, targets in val_loader:
                seqs, s_idxs, r_idxs, targets = seqs.to(cfg.device), s_idxs.to(cfg.device), r_idxs.to(cfg.device), targets.to(cfg.device)
                
                preds = model(seqs, s_idxs, r_idxs)
                
                # 1. Tính Weighted loss
                loss_w = criterion(preds, targets)
                val_weighted_loss_sum += loss_w.item()
                
                # 2. Tính Standard MSE
                loss_mse = torch.nn.functional.mse_loss(preds, targets)
                val_std_mse_sum += loss_mse.item()
        
        avg_val_weighted = val_weighted_loss_sum / len(val_loader)
        avg_val_std_mse = val_std_mse_sum / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']

        if epoch >= warmup_epochs:
            scheduler.step(avg_val_weighted)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_weighted)
        print(f"Ep {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val W-Loss: {avg_val_weighted:.4f} | Val MSE: {avg_val_std_mse:.4f} | LR: {current_lr:.6f}", end="")
        
        if avg_val_weighted < best_val_loss:
            best_val_loss = avg_val_weighted
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), cfg.save_model_path)
            print(" | Saved Best")
        else:
            patience_counter += 1
            print(f" | Pat: {patience_counter}/{cfg.patience}")
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping tại Epoch {epoch+1}")
                break
                
    print(f"Model saved: {cfg.save_model_path}")
    model.load_state_dict(torch.load(cfg.save_model_path))
    test_actuals, test_preds = evaluate_test_set(model, test_loader, cfg.device)
    
    return (model, train_loader, val_loader, test_loader, criterion,
            train_loss_history, val_loss_history, best_epoch, best_val_loss, 
            test_actuals, test_preds)
    
if __name__ == "__main__":
    (model, train_loader, val_loader, test_loader, criterion, 
     t_hist, v_hist, best_ep, best_loss, 
     test_act, test_pred) = run_training()
    
    # 1. Vẽ biểu đồ Training History & Test Result
    if t_hist and len(t_hist) > 0:
        print_final_report(best_loss, best_ep, cfg.epochs)
        save_results_to_pkl(t_hist, v_hist, best_ep, best_loss)
        plot_learning_curves(t_hist, v_hist)
        
        if len(test_act) > 0:
            print("Đang vẽ biểu đồ Test...")
            plot_prediction_comparison(
                torch.tensor(test_act), 
                torch.tensor(test_pred), 
                station_name="FINAL_TEST_RESULT"
            )
            
    # 2. Chạy phân tích nâng cao (Nếu bạn muốn dùng các hàm đã comment)
    # Vì giờ bạn đã có đủ train_loader, test_loader, criterion nên có thể uncomment đoạn này:
    
    # from model.data_analysis import analyze_data_distribution
    # analyze_data_distribution(train_loader, val_loader, test_loader)
    
    # from model.unified_evaluation import compare_datasets
    # compare_datasets(model, train_loader, val_loader, test_loader, cfg.device, criterion)