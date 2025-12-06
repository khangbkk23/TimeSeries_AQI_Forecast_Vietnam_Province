import os
import load_data as data
import clustering
import visualize
from sklearn.metrics import silhouette_score

def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    root_folder = "./data" 

    # Kiểm tra đường dẫn có tồn tại không
    if not os.path.exists(root_folder):
        print(f"LỖI: Đường dẫn '{root_folder}' không tồn tại! Hãy kiểm tra lại.")
        return

    # --- BƯỚC 1: LOAD DỮ LIỆU ---
    print("\n--- 1. Loading Data ---")
    df_train, df_val, df_test = data.load_train_val_test(root_folder)
    
    if df_train.empty:
        print("Lỗi: Tập Train rỗng. Kiểm tra lại thư mục 'train'.")
        return

    print(f"Số lượng mẫu Train: {len(df_train)}")
    print(f"Số lượng mẫu Val:   {len(df_val)}")
    print(f"Số lượng mẫu Test:  {len(df_test)}")

    # --- BƯỚC 2: TIỀN XỬ LÝ (Fit trên Train) ---
    print("\n--- 2. Preprocessing ---")
    X_train, X_val, X_test, scaler = data.prepare_data(df_train, df_val, df_test)

    # --- BƯỚC 3: KHÁM PHÁ (Elbow) ---
    print("\n--- 3. Running Elbow Method (Mining Phase) ---")
    # Chỉ chạy Elbow trên tập Train để tìm cấu trúc
    wcss = clustering.run_elbow_method(X_train, k_range=range(1, 11))
    visualize.plot_elbow(range(1, 11), wcss)
    
    # --- BƯỚC 4: HUẤN LUYỆN (Deployment Phase) ---
    print("\n--- 4. Training K-Means ---")
    kmeans_model, labels, centers = clustering.train_kmeans(X_train, k=3)
    
    # Vẽ heatmap tâm cụm để so sánh với bảng quy chuẩn
    visualize.plot_centroids_heatmap(kmeans_model, scaler, data.FEATURES)

    # --- BƯỚC 5: ĐÁNH GIÁ (Evaluation) ---
    print("\n--- 5. Evaluating on Val and Test Set ---")
    if X_val is not None:
        val_labels = kmeans_model.predict(X_val)
        val_score = silhouette_score(X_val, val_labels)
        print(f">> Silhouette Score trên tập Validation: {val_score:.4f}")
    else:
        print("Không có dữ liệu Validation.")
    if X_test is not None:
        # Dự đoán nhãn cho tập Test
        test_labels = kmeans_model.predict(X_test)
        
        # Tính điểm Silhouette
        sil_score = silhouette_score(X_test, test_labels)
        print(f">> Silhouette Score trên tập Test: {sil_score:.4f}")
        
    else:
        print("Không có dữ liệu Test để đánh giá.")
    # 1. Vẽ PCA để xem độ tách biệt
    visualize.visualize_clusters_pca(X_train, labels, centers)

    # 2. Vẽ Boxplot để xem đặc điểm (Quan trọng để làm báo cáo)
    visualize.visualize_cluster_features(X_train, labels, data.FEATURES)

if __name__ == "__main__":
    main()