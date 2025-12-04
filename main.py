import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from model.data_utils import get_dataloaders
from model.model import DualEmbeddingLSTM
from model.configs import cfg
from model.visualize import plot_learning_curves, plot_prediction_comparison
def print_final_report(best_loss, best_epoch, total_epochs):
    print(f"{'Summary':^40}")
    print("="*40)
    print(f"Tổng số Epochs dự kiến : {total_epochs}")
    print(f"Epoch tốt nhất         : {best_epoch}")
    print(f"Loss tốt nhất          : {best_loss:.6f}")
    print("="*40 + "\n")

def save_results_to_pkl(train_hist, val_hist, best_epoch, best_loss, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_history.pkl")
    
    data = {
        "train_loss_history": train_hist,
        "val_loss_history": val_hist,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith('__')}
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
            seqs = seqs.to(device)
            s_idxs = s_idxs.to(device)
            r_idxs = r_idxs.to(device)
            targets = targets.to(device)
            
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
    
    print(f"Final test metrics):")
    print(f"   MSE:    {final_mse:.4f}")
    print(f"   RMSE:    {final_rmse:.4f}")
    print(f"   MAE:   {final_mae:.4f}")
    print("-" * 45)
    
    return actuals, predictions

def run_training():
    print(f"Device: {cfg.device}")
    print(f"Configuration: Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.learning_rate}")

    config_dict = {
        'info_path': cfg.info_path,
        'train_dir': cfg.train_dir,
        'val_dir': cfg.val_dir,
        'test_dir': cfg.test_dir,
        'sequence_length': cfg.sequence_length,
        'batch_size': cfg.batch_size
    }
    
    train_loader, val_loader, test_loader, data_info = get_dataloaders(config_dict)
    
    if data_info['input_dim'] == 0:
        print("Không tìm thấy dữ liệu đầu vào")
        return None, None, [], [], 0, 0

    model = DualEmbeddingLSTM(
        config=cfg,
        num_stations=data_info['num_stations'],
        num_regions=data_info['num_regions'],
        input_dim=data_info['input_dim']
    ).to(cfg.device)
    
    print(f"Mô hình đã khởi tạo: {data_info['num_stations']} trạm, {data_info['num_regions']} vùng.")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        for seqs, s_idxs, r_idxs, targets in train_loader:
            seqs = seqs.to(cfg.device)
            s_idxs = s_idxs.to(cfg.device)
            r_idxs = r_idxs.to(cfg.device)
            targets = targets.to(cfg.device)
            
            optimizer.zero_grad()
            predictions = model(seqs, s_idxs, r_idxs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, s_idxs, r_idxs, targets in val_loader:
                seqs = seqs.to(cfg.device)
                s_idxs = s_idxs.to(cfg.device)
                r_idxs = r_idxs.to(cfg.device)
                targets = targets.to(cfg.device)
                
                predictions = model(seqs, s_idxs, r_idxs)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}", end="")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), cfg.save_model_path)
            print(" | Saved Best Model")
        else:
            patience_counter += 1
            print(f" | Patience {patience_counter}/{cfg.patience}")
            
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping kích hoạt tại Epoch {epoch+1}")
                break
                
    print(f"Mô hình tốt nhất được lưu tại: {cfg.save_model_path}")
    
    best_model_state = torch.load(cfg.save_model_path)
    model.load_state_dict(best_model_state)
    print("Đang tải lại model tốt nhất để kiểm tra trên tập Test...")
    test_actuals, test_preds = evaluate_test_set(model, test_loader, cfg.device)
    
    return model, val_loader, train_loss_history, val_loss_history, best_epoch, best_val_loss, test_actuals, test_preds

if __name__ == "__main__":
    model, val_loader, t_hist, v_hist, best_ep, best_loss, test_act, test_pred = run_training()
    
    if t_hist and len(t_hist) > 0:
        print_final_report(best_loss, best_ep, cfg.epochs)

        save_results_to_pkl(t_hist, v_hist, best_ep, best_loss)

        plot_learning_curves(t_hist, v_hist)
        
        if len(test_act) > 0:
            print("Đang vẽ biểu đồ kết quả trên tập Test...")
            
            t_act_tensor = torch.tensor(test_act)
            t_pred_tensor = torch.tensor(test_pred)
            
            plot_prediction_comparison(t_act_tensor, t_pred_tensor, station_name="FINAL_TEST_RESULT")