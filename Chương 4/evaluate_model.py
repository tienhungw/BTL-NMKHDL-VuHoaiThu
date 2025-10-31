import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import config 
from data_processor import load_and_split_data 

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():

    # 1. Tải dữ liệu Test
    print("\nBước 1: Tải dữ liệu Test")
    _, _, X_test, _, _, y_test, _, _ = load_and_split_data()
    
    if X_test is None:
        return 

    # 2. Tải mô hình đã huấn luyện
    print(f"Bước 2: Tải mô hình từ {config.MODEL_PATH}")
    try:
        model_pipeline = joblib.load(config.MODEL_PATH)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config.MODEL_PATH}.")
        return
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return
    
    print("Tải mô hình thành công.")

    # 3. Chạy dự đoán trên tập Test
    print("Bước 3: Chạy dự đoán trên tập Test")
    y_pred = model_pipeline.predict(X_test)
    print("Dự đoán hoàn tất.")

    # 4. Đánh giá mô hình
    print("\nBước 4: CÁC CHỈ SỐ ĐÁNH GIÁ MODEL (TRÊN TẬP TEST)")
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n  1. Root Mean Squared Error (RMSE): {rmse:,.0f} VND")
    print(f"     -> Như vậy trung bình mô hình dự đoán sai lệch khoảng {rmse:,.0f} VND so với giá thật.")

    r2 = r2_score(y_test, y_pred)
    print(f"\n  2. R-squared (R2): {r2:.4f}")
    print(f"     ->Mô hình giải thích được {r2*100:.2f}% sự biến thiên của giá nhà trên tập test.")
    
    print("\nSo sánh 10 dự đoán đầu tiên (trên tập Test):")
    comparison_df = pd.DataFrame({
        'Giá Thực Tế': y_test.values[:10], 
        'Giá Dự Đoán': y_pred[:10]
    })
    
    comparison_df['Giá Thực Tế'] = comparison_df['Giá Thực Tế'].map('{:,.0f} VND'.format)
    comparison_df['Giá Dự Đoán'] = comparison_df['Giá Dự Đoán'].map('{:,.0f} VND'.format)
    
    print(comparison_df.to_string())

if __name__ == "__main__":
    main()