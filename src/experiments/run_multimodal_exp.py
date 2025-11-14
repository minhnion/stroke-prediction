import argparse
import yaml
import os
import pandas as pd
import json
import logging
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.datasets import MultiModalStrokeDataset, get_transforms
from src.models.encoders import create_image_encoder, create_tabular_encoder
from src.models.multimodal.fusion_model import FusionModel
from src.trainers.multimodal_trainer import MultiModalTrainer
from src.trainers.callbacks import EarlyStopping
from src.metrics import get_binary_classification_metrics
from src.utils import setup_logging, plot_confusion_matrix, plot_roc_curve, plot_training_history

def run_experiment(model_config_path, data_config_path, trainer_config_path):
    with open(model_config_path, 'r') as f: model_config = yaml.safe_load(f)
    with open(data_config_path, 'r') as f: data_config = yaml.safe_load(f)
    with open(trainer_config_path, 'r') as f: trainer_config = yaml.safe_load(f)

    model_name_base = os.path.splitext(os.path.basename(model_config_path))[0]
    data_name_base = os.path.splitext(os.path.basename(data_config_path))[0]
    trainer_name_base = os.path.splitext(os.path.basename(trainer_config_path))[0]
    
    exp_name = f"{model_name_base}_{data_name_base}_{trainer_name_base}"
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
    logging.info(f"Model Config: {model_config_path}")
    logging.info(f"Data Config: {data_config_path}")
    logging.info(f"Trainer Config: {trainer_config_path}")

    logging.info("Preparing data...")
    full_df = pd.read_csv(data_config['csv_path'])

    if 'missing_values' in data_config:
        logging.info("Handling missing values...")
        for col, params in data_config['missing_values'].items():
            if params['strategy'] == 'mean':
                fill_value = full_df[col].mean()
            elif params['strategy'] == 'median':
                fill_value = full_df[col].median()
            elif params['strategy'] == 'mode':
                fill_value = full_df[col].mode()[0]
            else: 
                fill_value = params.get('value', 0)
            
            full_df[col].fillna(fill_value, inplace=True)
            logging.info(f"Filled missing values in '{col}' with strategy '{params['strategy']}' (value: {fill_value:.2f})")

    label_encoders = {}
    for col in data_config['categorical_features']:
        encoder = LabelEncoder()
        full_df[col] = encoder.fit_transform(full_df[col])
        label_encoders[col] = encoder

    train_val_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df[data_config['target_col']])
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df[data_config['target_col']])

    processed_dir = os.path.join('data/processed', data_config['dataset_name'])
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    
    img_transforms = get_transforms(data_config['image_size'], data_config['image_mean'], data_config['image_std'])

    dataset_params = {
        'image_root_dir': data_config['image_root_dir'],
        'image_path_col': data_config['image_path_col'],
        'categorical_cols': data_config['categorical_features'], 
        'numerical_cols': data_config['numerical_features'],   
        'target_col': data_config['target_col'],
        'transforms': img_transforms
    }

    train_dataset = MultiModalStrokeDataset(csv_path=os.path.join(processed_dir, 'train.csv'), **dataset_params)
    val_dataset = MultiModalStrokeDataset(csv_path=os.path.join(processed_dir, 'val.csv'), **dataset_params)

    train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, num_workers=data_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, num_workers=data_config['num_workers'])

    logging.info("Building model...")
    cat_dims = [len(full_df[col].unique()) for col in data_config['categorical_features']]
    model_config['tabular_encoder']['params']['categories'] = tuple(cat_dims)
    
    image_encoder, img_dim = create_image_encoder(**model_config['image_encoder'])
    tabular_encoder, tab_dim = create_tabular_encoder(**model_config['tabular_encoder'], data_config=data_config)
    
    model = FusionModel(image_encoder, tabular_encoder, img_dim, tab_dim, model_config['fusion'], model_config['mlp_head'])
    # device = torch.device(trainer_config['training_params']['device'] if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    logging.info(f"Using device: {device}")

    logging.info("Setting up training components...")
    optimizer = torch.optim.AdamW(model.parameters(), **trainer_config['optimizer']['params'])
    
    loss_cfg = trainer_config['loss_function']
    if loss_cfg['name'] == 'BCEWithLogitsLoss':
        pos_weight = None
        if loss_cfg['params'].get('use_pos_weight', False):
            neg = len(train_df) - train_df[data_config['target_col']].sum()
            pos = train_df[data_config['target_col']].sum()
            ratio = neg / pos
            if loss_cfg['params']['pos_weight_mode'] == 'sqrt':
                pos_weight = torch.tensor([torch.sqrt(torch.tensor(ratio))], device=device)
            else:
                pos_weight = torch.tensor([ratio], device=device)
            logging.info(f"Using pos_weight for BCE loss: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss function: {loss_cfg['name']}")

    metrics = get_binary_classification_metrics()
    early_stopping = EarlyStopping(patience=trainer_config['early_stopping']['patience'], verbose=True, path=os.path.join(checkpoints_dir, 'best_model.pth'))

    trainer = MultiModalTrainer(model, optimizer, criterion, device, train_loader, val_loader, metrics, trainer_config['training_params']['epochs'], early_stopping)
    
    history = trainer.train()

    with open(trainer_log_path, 'w') as f:
        json.dump(history, f, indent=4)
    logging.info(f"Trainer history saved to {trainer_log_path}")

    plot_training_history(history, os.path.join(plots_dir, 'training_history.png'))

    logging.info("Final evaluation on validation set with the best model...")
    y_true, y_pred_proba = trainer.evaluate(val_loader)
    y_pred = (y_pred_proba > 0.5).astype(int)

    final_metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1_score': f1_score(y_true, y_pred, zero_division=0), 'auroc': roc_auc_score(y_true, y_pred_proba)}
    
    logging.info(f"Final Validation Metrics: \n{json.dumps(final_metrics, indent=2)}")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    class_names = ['No Stroke', 'Stroke']
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_pred_proba, os.path.join(plots_dir, 'roc_curve.png'))

    logging.info(f"--- Experiment Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a flexible deep learning experiment.")
    parser.add_argument('--model', type=str, required=True, help='Path to the model config YAML file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data config YAML file.')
    parser.add_argument('--trainer', type=str, required=True, help='Path to the trainer config YAML file.')
    
    args = parser.parse_args()
    run_experiment(args.model, args.data, args.trainer)