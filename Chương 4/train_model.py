import xgboost as xgb
import joblib 
from sklearn.pipeline import Pipeline
import config 
from data_processor import load_and_split_data, create_preprocessor

def main():
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN")
    
    # 1. Tải dữ liệu 
    print("\nBước 1: Tải và xử lý dữ liệu")
    X_train, X_val, X_test, y_train, y_val, y_test, num_features, cat_features = load_and_split_data()
    
    if X_train is None:
        return 

    # 2. Tạo bộ tiền xử lý 
    print("Bước 2: Tạo bộ tiền xử lý (Preprocessor)")
    preprocessor = create_preprocessor(num_features, cat_features)

    # 3. Xử lý dữ liệu Train và Validate 
    print("Bước 3: Xử lý dữ liệu Train & Validate")
    print("Fitting preprocessor trên tập Train...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Transforming preprocessor trên tập Validate...")
    X_val_processed = preprocessor.transform(X_val)
    print("...Xử lý dữ liệu hoàn tất.")

    # 4. Khởi tạo mô hình (
    print("\nBước 4: Khởi tạo mô hình XGBoost")
    print("Các siêu tham số (Hyperparameters) được sử dụng:")
    for param, value in config.XGB_PARAMS.items():
        print(f"  - {param}: {value}")
    xgb_reg = xgb.XGBRegressor(**config.XGB_PARAMS)
    
    # 5. Huấn luyện mô hình 
    print("\nBước 5: Huấn luyện mô hình")
    print(f"Sử dụng Early Stopping với {config.XGB_PARAMS['early_stopping_rounds']} vòng.")
    eval_set = [(X_train_processed, y_train), (X_val_processed, y_val)]
    
    xgb_reg.fit(
        X_train_processed, 
        y_train,
        eval_set=eval_set,
        verbose=10 
    )
    print("...Huấn luyện hoàn tất!")

    # 6. Tạo và Lưu Pipeline hoàn chỉnh
    print(f"Bước 6: Lưu pipeline hoàn chỉnh vào {config.MODEL_PATH}")
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('regressor', xgb_reg)       
    ])
    joblib.dump(final_pipeline, config.MODEL_PATH)
    print("Đã lưu mô hình thành công.")
    
    print("\nQUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT")

if __name__ == "__main__":
    main()