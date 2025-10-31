import argparse
import yaml
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.models.ml_algorithm.xgboost_model import XGBoostModel

def run_ml_experiment(exp_config_path: str):
    with open(exp_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(config['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)
        
    print(f"--- Starting Experiment: {config['experiment_name']} ---")
        
    processed_dir = data_config['processed_path']
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv')).values.ravel()
    
    model = XGBoostModel(params=config['model']['params'])
    
    print("Training model...")
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=config['training']['early_stopping_rounds'])
    
    print("Evaluating model on validation set...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"  Val Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"  Val Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"  Val Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"  Val F1-score:  {f1_score(y_val, y_pred):.4f}")
    print(f"  Val AUROC:     {roc_auc_score(y_val, y_pred_proba):.4f}")
    
    print(f"--- Experiment Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an ML experiment from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config YAML file.')
    args = parser.parse_args()
    run_ml_experiment(args.config)