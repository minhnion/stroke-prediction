import argparse
import yaml
import os
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import StrokeDataset
from src.models.tabtransformer import TabTransformerModel # Giả sử tên file là transformer.py
from src.metrics import get_binary_classification_metrics
from src.trainers.trainer import Trainer

def run_experiment(exp_config_path: str):
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    with open(exp_config['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)

    device = torch.device(exp_config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processed_dir = data_config['processed_path']
    cat_cols = data_config['categorical_features']
    num_cols = data_config['numerical_features']
    
    train_dataset = StrokeDataset(
        features_path=os.path.join(processed_dir, 'X_train.csv'),
        labels_path=os.path.join(processed_dir, 'y_train.csv'),
        categorical_cols=cat_cols,
        numerical_cols=num_cols
    )
    val_dataset = StrokeDataset(
        features_path=os.path.join(processed_dir, 'X_val.csv'),
        labels_path=os.path.join(processed_dir, 'y_val.csv'),
        categorical_cols=cat_cols,
        numerical_cols=num_cols
    )
    
    train_loader = DataLoader(train_dataset, batch_size=exp_config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=exp_config['training']['batch_size'], shuffle=False)

    cat_cardinalities = joblib.load(os.path.join(processed_dir, 'cat_cardinalities.joblib'))
    
    model_params = exp_config['model']['params']
    model_params['categories'] = cat_cardinalities
    model_params['num_continuous'] = len(num_cols)
    
    model = TabTransformerModel(**model_params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), **exp_config['optimizer']['params'])
    
    y_train_df = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
    neg_count = (y_train_df['stroke'] == 0).sum()
    pos_count = (y_train_df['stroke'] == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics = get_binary_classification_metrics()
    
    checkpoint_dir = os.path.join(exp_config['results']['checkpoint_dir'], exp_config['experiment_name'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics,
        epochs=exp_config['training']['epochs'],
        checkpoint_dir=checkpoint_dir
    )

    print(f"--- Starting Experiment: {exp_config['experiment_name']} ---")
    trainer.train()
    print(f"--- Experiment Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a training experiment from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config YAML file.')
    
    args = parser.parse_args()
    run_experiment(args.config)