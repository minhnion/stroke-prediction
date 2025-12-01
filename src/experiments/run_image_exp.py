import argparse
import yaml
import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.datasets import ImageFolderWrapper, get_transforms
from src.models.encoders import create_image_encoder
from src.models.image.image_classifier import ImageClassifier
from src.trainers.multimodal_trainer import MultiModalTrainer
from src.trainers.callbacks import EarlyStopping
from src.metrics import get_binary_classification_metrics
from src.utils import setup_logging, plot_confusion_matrix, plot_roc_curve, plot_training_history, set_seed

def run_image_experiment(model_config_path, data_config_path, trainer_config_path):
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
    
    logging.info(f"--- Starting Image-Only Experiment: {exp_name} ---")
  
    seed = trainer_config.get('seed', 42)
    set_seed(seed)

    logging.info(f"Model Config: {model_config_path}")
    logging.info(f"Data Config: {data_config_path}")
    logging.info(f"Trainer Config: {trainer_config_path}")

    logging.info("Preparing image-only data from processed folder...")

    train_transforms = get_transforms(
        image_size=data_config['image_size'],
        mean=data_config['image_mean'],
        std=data_config['image_std'],
        augment=True
    )

    val_transforms = get_transforms(
        image_size=data_config['image_size'],
        mean=data_config['image_mean'],
        std=data_config['image_std'],
        augment=False
    )
    
    processed_image_dir = data_config['image_root_dir_processed']
    train_dir = os.path.join(processed_image_dir, 'train')
    val_dir = os.path.join(processed_image_dir, 'val')
    
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Processed data not found. Please run the image preprocessing script for '{data_config['dataset_name']}' first.")

    train_dataset = ImageFolderWrapper(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolderWrapper(root=val_dir, transform=val_transforms)
    
    logging.info(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    logging.info(f"Classes found: {train_dataset.classes} with mapping {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, num_workers=data_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, num_workers=data_config['num_workers'])

    logging.info(f"Building model: {model_config['model_name']}")
    image_encoder, img_dim = create_image_encoder(**model_config['image_encoder'])
    model = ImageClassifier(
        image_encoder=image_encoder,
        image_embedding_dim=img_dim,
        mlp_params=model_config['mlp_head']['params']
    )
    device = torch.device(trainer_config['training_params']['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}")

    logging.info("Setting up training components...")
    optimizer = torch.optim.AdamW(model.parameters(), **trainer_config['optimizer']['params'])
    
    scheduler = None
    if 'scheduler' in trainer_config:
        sched_cfg = trainer_config['scheduler']
        
        params = sched_cfg.get('params', {})
        
        if sched_cfg['name'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=trainer_config['training_params']['epochs'], 
                eta_min=float(params.get('eta_min', 1e-6)) 
            )
        elif sched_cfg['name'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(params.get('factor', 0.1)), 
                patience=int(params.get('patience', 5)), 
                min_lr=float(params.get('min_lr', 1e-6)) 
            )
        elif sched_cfg['name'] == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(params.get('T_0', 10)),          
                T_mult=int(params.get('T_mult', 1)),      
                eta_min=float(params.get('eta_min', 1e-6)) 
            )
        logging.info(f"Using Scheduler: {sched_cfg['name']}")


    loss_cfg = trainer_config['loss_function']

    if loss_cfg['name'] == 'BCEWithLogitsLoss':
        pos_weight = None
        if loss_cfg['params'].get('use_pos_weight', False):
            try:
                class_to_idx = train_dataset.class_to_idx
                logging.info(f"Calculating pos_weight from class mapping: {class_to_idx}")
                
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                
                neg_class_name = idx_to_class[0] 
                pos_class_name = idx_to_class[1] 
                
                neg_count = len(os.listdir(os.path.join(train_dir, neg_class_name)))
                pos_count = len(os.listdir(os.path.join(train_dir, pos_class_name)))
                
                logging.info(f"Count for '{neg_class_name}' (label 0): {neg_count}")
                logging.info(f"Count for '{pos_class_name}' (label 1): {pos_count}")

                if pos_count > 0:
                    ratio = neg_count / pos_count
                    if loss_cfg['params']['pos_weight_mode'] == 'sqrt':
                        pos_weight = torch.tensor([torch.sqrt(torch.tensor(ratio))], device=device)
                    else:
                        pos_weight = torch.tensor([ratio], device=device)
                    logging.info(f"Using pos_weight for BCE loss: {pos_weight.item():.2f}")
                else:
                    logging.warning("Positive class count is 0. Cannot calculate pos_weight.")

            except Exception as e:
                logging.warning(f"Could not calculate pos_weight for ImageFolder. Error: {e}. Proceeding without it.")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss function: {loss_cfg['name']}")

    metrics = get_binary_classification_metrics()
    early_stopping = EarlyStopping(
        patience=trainer_config['early_stopping']['patience'], 
        verbose=True, 
        path=os.path.join(checkpoints_dir, 'best_model.pth'),
        trace_func=logging.info
    )

    trainer = MultiModalTrainer(model, optimizer, criterion, device, train_loader, val_loader, metrics, trainer_config['training_params']['epochs'], early_stopping, scheduler=scheduler)
    
    history = trainer.train()

    with open(trainer_log_path, 'w') as f:
        json.dump(history, f, indent=4)
    logging.info(f"Trainer history saved to {trainer_log_path}")

    if history:
        plot_training_history(history, os.path.join(plots_dir, 'training_history.png'))

    logging.info("Final evaluation on validation set with the best model...")
    y_true, y_pred_proba = trainer.evaluate(val_loader)
    y_pred = (y_pred_proba > 0.5).astype(int)

    final_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_pred_proba)
    }
    
    logging.info(f"Final Validation Metrics: \n{json.dumps(final_metrics, indent=2)}")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    class_names = train_dataset.classes
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_pred_proba, os.path.join(plots_dir, 'roc_curve.png'))

    logging.info(f"--- Experiment Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an image-only classification experiment.")
    parser.add_argument('--model', type=str, required=True, help='Path to the model config YAML file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data config YAML file.')
    parser.add_argument('--trainer', type=str, required=True, help='Path to the trainer config YAML file.')
    
    args = parser.parse_args()
    run_image_experiment(args.model, args.data, args.trainer)