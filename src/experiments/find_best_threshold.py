import argparse
import yaml
import os
import pandas as pd
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.data.datasets import MultiModalStrokeDataset, get_transforms
from src.models.encoders import create_image_encoder, create_tabular_encoder
from src.models.multimodal.fusion_model import FusionModel
from src.utils import setup_logging

def find_best_threshold(model_config_path, data_config_path, trainer_config_path):
    # 1. Load Configs
    with open(model_config_path, 'r') as f: model_config = yaml.safe_load(f)
    with open(data_config_path, 'r') as f: data_config = yaml.safe_load(f)
    with open(trainer_config_path, 'r') as f: trainer_config = yaml.safe_load(f)

    model_name_base = os.path.splitext(os.path.basename(model_config_path))[0]
    data_name_base = os.path.splitext(os.path.basename(data_config_path))[0]
    trainer_name_base = os.path.splitext(os.path.basename(trainer_config_path))[0]
    exp_name = f"{model_name_base}_{data_name_base}_{trainer_name_base}"
    
    results_dir = os.path.join('results/experiments', exp_name)
    checkpoint_path = os.path.join(results_dir, 'checkpoints', 'best_model.pth')
    
    log_path = os.path.join('results/logs', f"{exp_name}_threshold_tuning.log")
    setup_logging(log_path)
    logging.info(f"--- Finding Best Threshold for: {exp_name} ---")

    # 2. Load Validation Data
    processed_dir = os.path.join('data/processed', data_config['dataset_name'])
    val_csv_path = os.path.join(processed_dir, 'val.csv')
    
    full_df = pd.read_csv(data_config['csv_path']) # Load để tính cat dims
    cat_dims = [len(full_df[col].unique()) for col in data_config['categorical_features']]
    model_config['tabular_encoder']['params']['categories'] = tuple(cat_dims)

    img_transforms = get_transforms(data_config['image_size'], data_config['image_mean'], data_config['image_std'])
    
    dataset_params = {
        'image_root_dir': data_config['image_root_dir'],
        'image_path_col': data_config['image_path_col'],
        'categorical_cols': data_config['categorical_features'], 
        'numerical_cols': data_config['numerical_features'],     
        'target_col': data_config['target_col'],
        'transforms': img_transforms
    }

    val_dataset = MultiModalStrokeDataset(
        csv_path=val_csv_path, 
        **dataset_params
    )
    val_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'] * 2, shuffle=False, num_workers=4)

    # 3. Load Model
    logging.info("Loading model...")
    image_encoder, img_dim = create_image_encoder(**model_config['image_encoder'])
    tabular_encoder, tab_dim = create_tabular_encoder(**model_config['tabular_encoder'], data_config=data_config)
    model = FusionModel(image_encoder, tabular_encoder, img_dim, tab_dim, model_config['fusion'], model_config['mlp_head'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Get Predictions (Probabilities)
    logging.info("Getting predictions on Validation set...")
    all_labels = []
    all_preds_proba = []
    
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(batch)
            preds_proba = torch.sigmoid(outputs)
            all_labels.append(batch['label'].cpu())
            all_preds_proba.append(preds_proba.cpu())

    y_true = torch.cat(all_labels).numpy().flatten()
    y_proba = torch.cat(all_preds_proba).numpy().flatten()

    # 5. Search for Best Threshold
    thresholds = np.arange(0.1, 0.95, 0.01)
    f1_scores = []
    precisions = []
    recalls = []
    accuracies = []

    best_f1 = 0
    best_thresh = 0.5

    logging.info("Scanning thresholds...")
    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        f1_scores.append(f1)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # 6. Report & Plot
    logging.info(f"\n>>> BEST THRESHOLD FOUND: {best_thresh:.2f}")
    logging.info(f"    Best Val F1-score: {best_f1:.4f}")
    
    # Tính lại các chỉ số khác tại ngưỡng tốt nhất
    y_pred_best = (y_proba > best_thresh).astype(int)
    logging.info(f"    Precision: {precision_score(y_true, y_pred_best):.4f}")
    logging.info(f"    Recall:    {recall_score(y_true, y_pred_best):.4f}")
    logging.info(f"    Accuracy:  {accuracy_score(y_true, y_pred_best):.4f}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='blue', linewidth=2)
    plt.plot(thresholds, precisions, label='Precision', color='green', linestyle='--')
    plt.plot(thresholds, recalls, label='Recall', color='red', linestyle='--')
    plt.axvline(best_thresh, color='orange', linestyle=':', label=f'Best Threshold ({best_thresh:.2f})')
    
    plt.title(f'Metrics vs. Decision Threshold\n(Model: {exp_name})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, 'plots', 'threshold_tuning.png')
    plt.savefig(plot_path)
    logging.info(f"Threshold plot saved to: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--trainer', required=True)
    args = parser.parse_args()
    find_best_threshold(args.model, args.data, args.trainer)