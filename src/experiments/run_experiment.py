# import argparse
# import yaml
# import os
# import joblib
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader

# from src.data.dataset import StrokeDataset
# from src.models.tabtransformer import TabTransformerModel # Giả sử tên file là transformer.py
# from src.metrics import get_binary_classification_metrics
# from src.trainers.trainer import Trainer
# from src.trainers.callbacks import EarlyStopping

# def run_experiment(exp_config_path: str):
#     with open(exp_config_path, 'r') as f:
#         exp_config = yaml.safe_load(f)
    
#     with open(exp_config['data_config_path'], 'r') as f:
#         data_config = yaml.safe_load(f)

#     device = torch.device(exp_config['training']['device'] if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     processed_dir = data_config['processed_path']
#     cat_cols = data_config['categorical_features']
#     num_cols = data_config['numerical_features']
    
#     train_dataset = StrokeDataset(
#         features_path=os.path.join(processed_dir, 'X_train.csv'),
#         labels_path=os.path.join(processed_dir, 'y_train.csv'),
#         categorical_cols=cat_cols,
#         numerical_cols=num_cols
#     )
#     val_dataset = StrokeDataset(
#         features_path=os.path.join(processed_dir, 'X_val.csv'),
#         labels_path=os.path.join(processed_dir, 'y_val.csv'),
#         categorical_cols=cat_cols,
#         numerical_cols=num_cols
#     )
    
#     train_loader = DataLoader(train_dataset, batch_size=exp_config['training']['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=exp_config['training']['batch_size'], shuffle=False)

#     cat_cardinalities = joblib.load(os.path.join(processed_dir, 'cat_cardinalities.joblib'))
    
#     model_params = exp_config['model']['params']
#     model_params['categories'] = cat_cardinalities
#     model_params['num_continuous'] = len(num_cols)
    
#     model = TabTransformerModel(**model_params).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), **exp_config['optimizer']['params'])
    
#     if data_config.get('apply_smote', False):
#         criterion = torch.nn.BCEWithLogitsLoss()
#     else:
#         y_train_df = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
#         neg_count = (y_train_df['stroke'] == 0).sum()
#         pos_count = (y_train_df['stroke'] == 1).sum()
#         pos_weight_ratio = neg_count / pos_count
#         pos_weight = torch.tensor([torch.sqrt(torch.tensor(pos_weight_ratio))], device=device)
        
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     metrics = get_binary_classification_metrics()
    
#     checkpoint_dir = os.path.join(exp_config['results']['checkpoint_dir'], exp_config['experiment_name'])

#     os.makedirs(checkpoint_dir, exist_ok=True)

#     early_stopping_params = exp_config.get('early_stopping', {}) 
#     early_stopping = EarlyStopping(
#         patience=early_stopping_params.get('patience', 10), 
#         verbose=True,
#         path=os.path.join(checkpoint_dir, 'best_model.pth')
#     )
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         criterion=criterion,
#         device=device,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         metrics=metrics,
#         epochs=exp_config['training']['epochs'],
#         early_stopping=early_stopping
#     )

#     print(f"--- Starting Experiment: {exp_config['experiment_name']} ---")
#     trainer.train()
#     print(f"--- Experiment Finished ---")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run a training experiment from a config file.")
#     parser.add_argument('--config', type=str, required=True, help='Path to the experiment config YAML file.')
    
#     args = parser.parse_args()
#     run_experiment(args.config)

# src/experiments/run_experiment.py

import argparse
import yaml
import os
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.dataset import StrokeDataset
from src.models.tabtransformer import TabTransformerModel
from src.metrics import get_binary_classification_metrics
from src.trainers.trainer import Trainer
from src.trainers.callbacks import EarlyStopping
from src.utils import setup_logging, plot_confusion_matrix, plot_roc_curve, plot_training_history

def run_experiment(exp_config_path: str):
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    with open(exp_config['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)

    exp_name = exp_config['experiment_name']
    
    # --- THIẾT LẬP CẤU TRÚC THƯ MỤC VÀ LOGGING ---
    results_dir = os.path.join('results/experiments', exp_name)
    plots_dir = os.path.join(results_dir, 'plots')
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    metrics_path = os.path.join(results_dir, 'metrics.json')
    trainer_log_path = os.path.join(results_dir, 'trainer_log.json')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    log_path = os.path.join('results/logs', f"{exp_name}.log")
    setup_logging(log_path)

    logging.info(f"--- Starting Experiment: {exp_name} ---")
    
    device = torch.device(exp_config['training']['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    processed_dir = data_config['processed_path']
    cat_cols = data_config['categorical_features']
    num_cols = data_config['numerical_features']
    
    train_dataset = StrokeDataset(features_path=os.path.join(processed_dir, 'X_train.csv'), labels_path=os.path.join(processed_dir, 'y_train.csv'), categorical_cols=cat_cols, numerical_cols=num_cols)
    val_dataset = StrokeDataset(features_path=os.path.join(processed_dir, 'X_val.csv'), labels_path=os.path.join(processed_dir, 'y_val.csv'), categorical_cols=cat_cols, numerical_cols=num_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=exp_config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=exp_config['training']['batch_size'], shuffle=False)

    cat_cardinalities = joblib.load(os.path.join(processed_dir, 'cat_cardinalities.joblib'))
    
    model_params = exp_config['model']['params']
    model_params['categories'] = cat_cardinalities
    model_params['num_continuous'] = len(num_cols)
    model = TabTransformerModel(**model_params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), **exp_config['optimizer']['params'])
    
    if data_config.get('use_smote', False):
        logging.info("Using standard BCEWithLogitsLoss (for SMOTE balanced data)")
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        logging.info("Using BCEWithLogitsLoss with pos_weight")
        y_train_df = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
        neg_count = (y_train_df['stroke'] == 0).sum()
        pos_count = (y_train_df['stroke'] == 1).sum()
        if pos_count > 0:
            pos_weight_ratio = neg_count / pos_count
            pos_weight = torch.tensor([torch.sqrt(torch.tensor(pos_weight_ratio))], device=device)
            logging.info(f"Calculated pos_weight: {pos_weight.item():.2f}")
        else:
            pos_weight = None
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics = get_binary_classification_metrics()
    
    early_stopping_params = exp_config.get('early_stopping', {})
    early_stopping = EarlyStopping(patience=early_stopping_params.get('patience', 10), verbose=True, path=os.path.join(checkpoints_dir, 'best_model.pth'))

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device, train_loader=train_loader, val_loader=val_loader, metrics=metrics, epochs=exp_config['training']['epochs'], early_stopping=early_stopping)

    history = trainer.train()

    # Lưu lịch sử huấn luyện
    with open(trainer_log_path, 'w') as f:
        json.dump(history, f, indent=4)
    logging.info(f"Trainer history saved to {trainer_log_path}")

    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(history, os.path.join(plots_dir, 'training_history.png'))

    logging.info("Final evaluation on validation set with the best model...")
    y_true, y_pred_proba = trainer.evaluate(val_loader)
    y_pred = (y_pred_proba > 0.5).astype(int)

    final_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_pred_proba)
    }

    logging.info("Final Validation Metrics:")
    for key, value in final_metrics.items():
        logging.info(f"  {key.capitalize()}: {value:.4f}")
    
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")
    
    class_names = ['No Stroke', 'Stroke']
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_pred_proba, os.path.join(plots_dir, 'roc_curve.png'))

    logging.info(f"--- Experiment Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config YAML file.')
    args = parser.parse_args()
    run_experiment(args.config)