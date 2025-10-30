import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os 
import joblib
import argparse

def preprocess_data(config: dict):
    df = pd.read_csv(config['raw_path'])
    df = df.drop(columns=config['drop_columns'])

    if 'missing_values' in config:
        for col, params in config['missing_values'].items():
            if params['strategy'] == 'mean':
                fill_value = df[col].mean()
            elif params['strategy'] == 'median':
                fill_value = df[col].median()
            elif params['strategy'] == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = params.get('value', 0)
            df[col].fillna(fill_value, inplace=True)
    
    categorical_cols = config['categorical_features']
    numerical_cols = config['numerical_features']
    target_col = config['target_col']

    label_encoders = {}
    cat_cardinalities = []
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        label_encoders[col] = encoder
        cat_cardinalities.append(len(encoder.classes_))

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    split_params = config['split_params']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=split_params['test_size'],
        random_state=split_params['random_state'],
        stratify=y
    )
    
    val_size_adjusted = split_params['val_size'] / (1 - split_params['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=split_params['random_state'],
        stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    processed_dir = config['processed_path']
    os.makedirs(processed_dir, exist_ok=True) 
    
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    X_val.to_csv(os.path.join(processed_dir, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(processed_dir, 'y_val.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    print(f"Đã lưu các tập dữ liệu vào thư mục: {processed_dir}")

    joblib.dump(scaler, os.path.join(processed_dir, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(processed_dir, 'label_encoders.joblib'))
    joblib.dump(cat_cardinalities, os.path.join(processed_dir, 'cat_cardinalities.joblib'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data preprocessing from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the data config YAML file.')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    preprocess_data(config_dict)