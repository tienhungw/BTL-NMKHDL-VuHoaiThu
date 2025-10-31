# 1. Cấu hình file
FILE_PATH = "D:/Khoa_hoc_du_lieu/BTN/chuong4/clean_data_F5.csv" 
MODEL_PATH = "xgboost_pipeline.joblib" 

# 2. Cấu hình cho features
TARGET = 'Giá (VND)'

DROP_COLS = ['Ngày đăng', 'Giá_MinMax', 'Diện tích_MinMax', 'Giá_Zscore']

# Cột categorical 
CATEGORICAL_FEATURES = ['Loại hình căn hộ', 'Tình trạng nội thất', 'Quận']

# 3. Cấu hình chia dữ liệu 
TEST_SIZE = 0.15        
VALIDATION_SIZE = 0.15  
RANDOM_STATE = 42

# 4. Cấu hình siêu tham số XGBoost
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 10000,           
    'learning_rate': 0.001,          
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'early_stopping_rounds': 10
}