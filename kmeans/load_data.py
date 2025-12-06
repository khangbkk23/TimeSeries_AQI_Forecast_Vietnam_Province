import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Cấu hình các cột đặc trưng (Feature)
FEATURES = ['CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2'] 

def load_folder(folder_path):
    """
    Đọc tất cả file csv trong folder và chỉ lấy các cột FEATURES.
    Không cần inverse, không cần quan tâm scaler cũ.
    """
    search_path = os.path.join(folder_path, "*.csv")
    all_files = glob.glob(search_path)
    
    if len(all_files) == 0:
        print(f"CẢNH BÁO: Không tìm thấy file .csv nào trong '{folder_path}'")
        return pd.DataFrame()

    df_list = []
    
    print(f"Đang đọc {len(all_files)} files từ: {os.path.basename(folder_path)}")

    for filename in all_files:
        try:
            # 1. Đọc file
            df = pd.read_csv(filename)
            
            if df.empty: continue

            # 2. Kiểm tra cột
            # Chỉ lấy đúng các cột features cần thiết
            missing_cols = [col for col in FEATURES if col not in df.columns]
            
            if missing_cols:
                # Nếu thiếu cột quan trọng thì bỏ qua file này hoặc báo lỗi
                print(f"File {os.path.basename(filename)} thiếu cột: {missing_cols} -> Bỏ qua.")
                continue

            # 3. Lấy dữ liệu
            df_subset = df[FEATURES]
            df_list.append(df_subset)
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {os.path.basename(filename)}: {e}")

    if not df_list:
        return pd.DataFrame()

    # 4. Gộp tất cả lại
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df

def load_train_val_test(root_folder):
    """
    Hàm chính để load dữ liệu cho 3 tập.
    Không cần tham số scaler_path nữa vì ta không dùng lại scaler cũ.
    """
    print(f"Đang đọc dữ liệu từ: {root_folder}")
    
    # Load từng folder
    print("--- Xử lý tập Train ---")
    df_train = load_folder(os.path.join(root_folder, "train"))
    
    print("--- Xử lý tập Validation ---")
    df_val = load_folder(os.path.join(root_folder, "validation")) 
    
    print("--- Xử lý tập Test ---")
    df_test = load_folder(os.path.join(root_folder, "test"))
    
    return df_train, df_val, df_test

def prepare_data(df_train, df_val, df_test):
    """
    Đầu vào: Dữ liệu đã Standard Scaled (mean=0, std=1).
    Nhiệm vụ: Chỉ xử lý ngoại lai trên tập Train.
    Đầu ra: Numpy arrays (X_train, X_val, X_test) và None (thay cho scaler).
    """

    # 1. Copy dữ liệu để không ảnh hưởng dataframe gốc
    df_train_clean = df_train.copy()

    # --- BƯỚC 1: XỬ LÝ NGOẠI LAI (Chỉ áp dụng trên tập Train) ---
    if not df_train_clean.empty:
        # Tính IQR
        Q1 = df_train_clean[FEATURES].quantile(0.25)
        Q3 = df_train_clean[FEATURES].quantile(0.75)
        IQR = Q3 - Q1

        # Xác định cận dưới và trên
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Lọc bỏ các dòng chứa ngoại lai
        # Vì dữ liệu đã Standard Scaled, ngoại lai thường là các giá trị > 3 hoặc < -3 (tùy phân phối)
        condition = ~((df_train_clean[FEATURES] < lower_bound) | (df_train_clean[FEATURES] > upper_bound)).any(axis=1)
        
        n_before = len(df_train_clean)
        df_train_clean = df_train_clean[condition]
        n_after = len(df_train_clean)
        
        print(f"Đã loại bỏ {n_before - n_after} dòng ngoại lai trên tập Train.")
    else:
        raise ValueError("Tập Train rỗng!")

    # --- BƯỚC 2: CHUYỂN ĐỔI SANG NUMPY ARRAY ---
    # Không scale nữa, chỉ lấy values
    
    X_train = df_train_clean[FEATURES].values
    
    # Val và Test giữ nguyên (kể cả ngoại lai) để đánh giá thực tế
    X_val = df_val[FEATURES].values if not df_val.empty else None
    X_test = df_test[FEATURES].values if not df_test.empty else None

    # Trả về None ở vị trí cuối cùng để thay thế cho 'scaler' (giữ code gọi hàm không bị lỗi unpack)
    return X_train, X_val, X_test, None