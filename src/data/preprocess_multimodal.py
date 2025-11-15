import argparse
import yaml
import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils import setup_logging

def preprocess_data(data_config_path: str):
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    log_path = os.path.join('results/logs', f"preprocess_{data_config['dataset_name']}.log")
    setup_logging(log_path)

    logging.info(f"--- Starting Preprocessing for Dataset: {data_config['dataset_name']} ---")
    
    logging.info(f"Loading raw data from {data_config['csv_path']}")
    full_df = pd.read_csv(data_config['csv_path'])
    
    if 'missing_values' in data_config:
        logging.info("Handling missing values...")
        for col, params in data_config['missing_values'].items():
            if params['strategy'] == 'mean':
                fill_value = full_df[col].mean()
            elif params['strategy'] == 'median':
                fill_value = full_df[col].median()
            else:
                fill_value = full_df[col].mode()[0]
            
            full_df[col] = full_df[col].fillna(fill_value)
            logging.info(f"Filled missing values in '{col}' with strategy '{params['strategy']}' (value: {fill_value:.2f})")
    
    logging.info("Applying Label Encoding to categorical features...")
    for col in data_config['categorical_features']:
        encoder = LabelEncoder()
        full_df[col] = encoder.fit_transform(full_df[col])

    logging.info("Splitting data into train, validation, and test sets...")
    train_val_df, test_df = train_test_split(
        full_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=full_df[data_config['target_col']]
    )
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.1, 
        random_state=42, 
        stratify=train_val_df[data_config['target_col']]
    )
    logging.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    processed_dir = os.path.join('data/processed', data_config['dataset_name'])
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    
    logging.info(f"Preprocessed data splits saved to {processed_dir}")
    logging.info("--- Preprocessing Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess a multi-modal dataset from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the data config file.')
    args = parser.parse_args()
    preprocess_data(args.config)