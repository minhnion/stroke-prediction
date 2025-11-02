import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

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