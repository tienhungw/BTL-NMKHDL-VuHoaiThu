import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import config 
import re 

def detect_outliers_by_cols(df, cols_to_check):
    outlier_indices = set()
    for col in cols_to_check:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            min_whisker = Q1 - 1.5*IQR
            max_whisker = Q3 + 1.5*IQR
            outliers = df[(df[col] < min_whisker) | (df[col] > max_whisker)].index
            outlier_indices.update(outliers)
    return list(outlier_indices)

def extract_district(address_str):
    try:
        parts = str(address_str).split(',')
        if len(parts) > 1:
            district = parts[-2].strip()
            if 'Quận' in district or 'Huyện' in district:
                return district
    except Exception as e:
        pass 
    return 'Không rõ' 

# Trích xuất đặc trưng từ cột 'Tiêu đề'
def extract_title_features(df):
    print("Trích xuất đặc trưng từ 'Tiêu đề'...")
    # Chuẩn hóa cột tiêu đề: chuyển sang chữ thường và fill NaN
    df['Tiêu đề_norm'] = df['Tiêu đề'].fillna('').astype(str).str.lower()

    # Định nghĩa các nhóm từ khóa "vàng"
    keywords_map = {
        'feature_oto': ['ô tô', 'oto', 'xe hơi', 'gara', 'ô tô đỗ', 'ô tô tránh'],
        'feature_kinh_doanh': ['kinh doanh', 'cho thuê', 'văn phòng', 'kd'],
        'feature_mat_pho': ['mặt phố', 'mặt tiền', 'mp', 'vip', 'phố', 'đường lớn'],
        'feature_lo_goc': ['lô góc'],
        'feature_thang_may': ['thang máy', 'thang máy xịn'],
        'feature_ngo': ['ngõ', 'hẻm', 'ngách']
    }
    
    new_features_list = []
    for feature_name, keywords in keywords_map.items():
        # Tạo pattern regex từ danh sách từ khóa
        pattern = '|'.join([re.escape(kw) for kw in keywords])
        # Tạo cột mới (1 nếu có, 0 nếu không)
        df[feature_name] = df['Tiêu đề_norm'].str.contains(pattern, regex=True).astype(int)
        new_features_list.append(feature_name)
    
    print(f"...Đã tạo {len(new_features_list)} đặc trưng mới từ Tiêu đề.")
    
    df = df.drop(columns=['Tiêu đề_norm'])
    return df


def load_and_split_data():
    try:
        df = pd.read_csv(config.FILE_PATH)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {config.FILE_PATH}")
        return (None,) * 8 

    print(f"Số dòng ban đầu (trước khi xử lý): {len(df)}")

    # --- Bước 1: Loại bỏ ngoại lai (Giữ nguyên) ---
    cols_to_check_outliers = ['Giá (VND)', 'Diện tích (m2)']
    outlier_indices = detect_outliers_by_cols(df, cols_to_check_outliers)
    print(f"Phát hiện {len(outlier_indices)} dòng ngoại lai (theo Giá & Diện tích).")
    if len(outlier_indices) > 0:
        df = df.drop(outlier_indices).reset_index(drop=True) 
        print(f"Số dòng còn lại (sau khi loại bỏ ngoại lai): {len(df)}")
    
    # --- Bước 2: Trích xuất 'Quận' (Giữ nguyên) ---
    df['Quận'] = df['Địa chỉ'].apply(extract_district)
    
    # --- Trích xuất 'Tiêu đề' ---
    df = extract_title_features(df)

    # --- Cập nhật danh sách features ---
    existing_drop_cols = [col for col in config.DROP_COLS if col in df.columns]
    
    features = [col for col in df.columns if col not in [config.TARGET] + existing_drop_cols + ['Địa chỉ', 'Tiêu đề']]
    
    cat_features = [col for col in config.CATEGORICAL_FEATURES if col in features]
    num_features = [col for col in features if col not in cat_features]
    
    print("\nDanh sách Features")
    print(f"Numerical features: {num_features}")
    print(f"Categorical features: {cat_features}")
    
    X = df[features].copy()
    y = df[config.TARGET].copy()
    
    # Chia dữ liệu thành 3 tập: Train, Validate, Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    if len(X_train_full) == 0:
        print("Lỗi: Không còn đủ dữ liệu.")
        return (None,) * 8

    val_split_ratio = config.VALIDATION_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_split_ratio, random_state=config.RANDOM_STATE
    )
    
    print(f"\nĐã tải và chia dữ liệu:")
    print(f"  - Tập Train:    {X_train.shape[0]} mẫu")
    print(f"  - Tập Validate: {X_val.shape[0]} mẫu")
    print(f"  - Tập Test:     {X_test.shape[0]} mẫu")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_features, cat_features


def create_preprocessor(num_features, cat_features):
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Không rõ')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor