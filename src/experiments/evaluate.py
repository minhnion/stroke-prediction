import argparse
import yaml
import os
import pandas as pd
import json
import logging
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.dataset import StrokeDataset
from src.models.tabtransformer import TabTransformerModel
from src.models.ml_algorithm.xgboost_model import XGBoostModel
from src.models.ml_algorithm.rf_model import RandomForestModel
from src.models.ml_algorithm.lightgbm_model import LightGBMModel
from src.models.ml_algorithm.catboost_model import CatBoostModel
from src.models.ml_algorithm.svm_model import SVMModel
from src.utils import setup_logging, plot_confusion_matrix, plot_roc_curve

def evaluate_model(experiment_name: str):
    exp_config_path = os.path.join('configs/experiments', f"{experiment_name}.yaml")
    with open(exp_config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)

    results_dir = os.path.join('results/experiments', experiment_name)
    test_eval_dir = os.path.join(results_dir, 'test_evaluation') # Thư mục mới cho kết quả test
    metrics_path = os.path.join(test_eval_dir, 'test_metrics.json')
    plots_dir = test_eval_dir
    os.makedirs(plots_dir, exist_ok=True)
    
    log_path = os.path.join('results/logs', f"{experiment_name}_test_eval.log")
    setup_logging(log_path)
    
    logging.info(f"--- Starting Final Evaluation on TEST SET for: {experiment_name} ---")

    processed_dir = data_config['processed_path']
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
    logging.info(f"Test data loaded. Shape: {X_test.shape}")

    model_name = config['model']['name']
    exp_type = config['experiment_type']

    y_pred = None
    y_pred_proba = None

    if exp_type == 'ml':
        checkpoint_path = os.path.join(results_dir, 'model_checkpoint.joblib')
        logging.info(f"Loading ML model from {checkpoint_path}")
        model = joblib.load(checkpoint_path)

        if model_name == "SVMModel":
            logging.info("Applying One-Hot Encoding for SVM...")
            categorical_cols = data_config['categorical_features']
            
            X_train_ref = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
            X_train_ref_ohe = pd.get_dummies(X_train_ref, columns=categorical_cols, dummy_na=False)

            X_test = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=False)
            X_test = X_test.reindex(columns=X_train_ref_ohe.columns, fill_value=0)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    elif exp_type == 'dl':
        checkpoint_path = os.path.join(results_dir, 'checkpoints', 'best_model.pth')
        logging.info(f"Loading DL model from {checkpoint_path}")
        
        device = torch.device("cpu") 
        
        cat_cols = data_config['categorical_features']
        num_cols = data_config['numerical_features']
        cat_cardinalities = joblib.load(os.path.join(processed_dir, 'cat_cardinalities.joblib'))
        
        model_params = config['model']['params']
        model_params['categories'] = cat_cardinalities
        model_params['num_continuous'] = len(num_cols)

        if model_name == "TabTransformerModel":
            model = TabTransformerModel(**model_params).to(device)
        else:
            raise ValueError(f"Unknown DL model: {model_name}")
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        test_dataset = StrokeDataset(features_path=os.path.join(processed_dir, 'X_test.csv'), labels_path=os.path.join(processed_dir, 'y_test.csv'), categorical_cols=cat_cols, numerical_cols=num_cols)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'] * 2, shuffle=False)
        
        all_preds_proba = []
        with torch.no_grad():
            for x_cat, x_cont, _ in test_loader:
                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                outputs = model(x_cat, x_cont)
                preds_proba = torch.sigmoid(outputs)
                all_preds_proba.append(preds_proba.cpu())
        
        y_pred_proba = torch.cat(all_preds_proba).numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

    logging.info("Calculating final metrics on test set...")
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_pred_proba)
    }

    logging.info("Final Test Metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key.capitalize()}: {value:.4f}")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Test metrics saved to {metrics_path}")

    class_names = ['No Stroke', 'Stroke']
    plot_confusion_matrix(y_test, y_pred, class_names, os.path.join(plots_dir, 'test_confusion_matrix.png'))
    plot_roc_curve(y_test, y_pred_proba, os.path.join(plots_dir, 'test_roc_curve.png'))

    logging.info(f"--- Final Evaluation Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument('--name', type=str, required=True, help='The name of the experiment to evaluate (e.g., xgboost_tuned).')
    args = parser.parse_args()
    evaluate_model(args.name)