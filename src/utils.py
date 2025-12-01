import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import contextlib
import joblib
from tqdm import tqdm

import random
import numpy as np
import torch

def setup_logging(log_path):
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    logging.info(f'Confusion matrix saved to {save_path}')

def plot_roc_curve(y_true, y_pred_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f'ROC curve saved to {save_path}')

def plot_training_history(history, save_path):
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training History', fontsize=16)

    # 1. Loss vs. Epoch
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs. Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. F1-score vs. Epoch
    axes[0, 1].plot(df['epoch'], df['val_f1_score'], label='Validation F1-score', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-score')
    axes[0, 1].set_title('F1-score vs. Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Precision & Recall vs. Epoch
    axes[1, 0].plot(df['epoch'], df['val_precision'], label='Validation Precision', color='red')
    axes[1, 0].plot(df['epoch'], df['val_recall'], label='Validation Recall', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall vs. Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. AUROC vs. Epoch
    axes[1, 1].plot(df['epoch'], df['val_auroc'], label='Validation AUROC', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUROC')
    axes[1, 1].set_title('AUROC vs. Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Training history plot saved to {save_path}")

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback


def load_weights(model, checkpoint_path, model_type):

    logging.info(f"Smart loading weights for {model_type} from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logging.error(f"Failed to load checkpoint file: {e}")
        raise e
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
    else:
        state_dict = checkpoint

    mapping_rules = {}
    
    # --- ResNet50 ---
    if 'resnet' in model_type.lower():
        mapping_rules = {
            'backbone.0.': 'conv1.', 
            'backbone.1.': 'bn1.',
            # backbone.2 (relu) và 3 (maxpool) không có tham số
            'backbone.4.': 'layer1.', 
            'backbone.5.': 'layer2.',
            'backbone.6.': 'layer3.', 
            'backbone.7.': 'layer4.'
        }
        
    # --- DenseNet121 ---
    elif 'densenet' in model_type.lower():
        mapping_rules = {
            'backbone.0.': 'features.', 
        }
        
    # --- InceptionV3  ---
    elif 'inception' in model_type.lower():
        mapping_rules = {
            'backbone.0.': '', 
        }

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:] 
        
        new_key = k
        for old, new in mapping_rules.items():
            if k.startswith(old):
                new_key = k.replace(old, new)
                break
        
        if 'fc.' in new_key or 'classifier.' in new_key or 'classif.' in new_key:
            continue
            
        new_state_dict[new_key] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    
    missing_keys = msg.missing_keys
    real_missing = [k for k in missing_keys if not any(x in k for x in ['fc', 'classifier', 'head'])]
    
    total_params = len(list(model.parameters()))
    loaded_params = len(new_state_dict)
    
    logging.info(f"Checkpoint keys processed: {loaded_params}")
    
    if len(real_missing) > 20: 
        logging.warning(f"⚠️ HIGH RISK: {len(real_missing)} layers were NOT loaded! Model might be random.")
        logging.warning(f"Sample missing keys: {real_missing[:5]}")
    else:
        logging.info(f"✅ SUCCESS: Weights loaded correctly. (Ignored head layers: {len(missing_keys) - len(real_missing)})")
    
    return model