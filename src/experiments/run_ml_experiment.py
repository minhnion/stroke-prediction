import argparse
import yaml
import os
import pandas as pd
import json
import logging
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.models.ml_algorithm.xgboost_model import XGBoostModel
from src.utils import setup_logging, plot_confusion_matrix, plot_roc_curve

def run_ml_experiment(exp_config_path: str):
    with open(exp_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(config['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)
        
    exp_name = config['experiment_name']


    results_dir = os.path.join('results/experiments', exp_name)
    plots_dir = os.path.join(results_dir, 'plots')
    checkpoint_path = os.path.join(results_dir, 'model_checkpoint.joblib')
    metrics_path = os.path.join(results_dir, 'metrics.json')
    os.makedirs(plots_dir, exist_ok=True)

    log_path = os.path.join('results/logs', f"{exp_name}.log")
    setup_logging(log_path)
    
    logging.info(f"--- Starting Experiment: {exp_name} ---")

    processed_dir = data_config['processed_path']
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv')).values.ravel()
    
    model = XGBoostModel(params=config['model']['params'])
    
    logging.info("Training model...")
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=config['training']['early_stopping_rounds'])
    
    logging.info("Saving the best model...")
    joblib.dump(model.model, checkpoint_path)
    logging.info("Evaluating model on validation set...")

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'auroc': roc_auc_score(y_val, y_pred_proba)
    }
    
    logging.info("Final Validation Metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key.capitalize()}: {value:.4f}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

    class_names = ['No Stroke', 'Stroke']
    plot_confusion_matrix(y_val, y_pred, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_val, y_pred_proba, os.path.join(plots_dir, 'roc_curve.png'))
    
    logging.info(f"--- Experiment Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an ML experiment from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config YAML file.')
    args = parser.parse_args()
    run_ml_experiment(args.config)