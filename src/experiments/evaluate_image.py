import argparse
import yaml
import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.data.datasets import ImageFolderWrapper, get_transforms
from src.models.encoders import create_image_encoder
from src.models.image.image_classifier import ImageClassifier
from src.utils import setup_logging, plot_confusion_matrix, plot_roc_curve

def evaluate_image_on_test(model_config_path, data_config_path, trainer_config_path):
    with open(model_config_path, 'r') as f: model_config = yaml.safe_load(f)
    with open(data_config_path, 'r') as f: data_config = yaml.safe_load(f)
    with open(trainer_config_path, 'r') as f: trainer_config = yaml.safe_load(f)

    model_name_base = os.path.splitext(os.path.basename(model_config_path))[0]
    data_name_base = os.path.splitext(os.path.basename(data_config_path))[0]
    trainer_name_base = os.path.splitext(os.path.basename(trainer_config_path))[0]
    
    exp_name = f"{model_name_base}_{data_name_base}_{trainer_name_base}"
    results_dir = os.path.join('results/experiments', exp_name)
    test_eval_dir = os.path.join(results_dir, 'test_evaluation')
    metrics_path = os.path.join(test_eval_dir, 'test_metrics.json')
    plots_dir = test_eval_dir
    os.makedirs(plots_dir, exist_ok=True)
    
    log_path = os.path.join('results/logs', f"{exp_name}_test_eval.log")
    setup_logging(log_path)
    
    logging.info(f"--- Starting Final Image-Only Evaluation on TEST SET for: {exp_name} ---")

    processed_image_dir = data_config['image_root_dir_processed']
    test_dir = os.path.join(processed_image_dir, 'test')

    if not os.path.isdir(test_dir):
        logging.error(f"Test data not found at {test_dir}. Please run the image preprocessing script first.")
        return

    img_transforms = get_transforms(data_config['image_size'], data_config['image_mean'], data_config['image_std'])
    
    test_dataset = ImageFolderWrapper(root=test_dir, transform=img_transforms)
    test_loader = DataLoader(test_dataset, batch_size=data_config['batch_size'] * 2, shuffle=False, num_workers=data_config['num_workers'])
    logging.info(f"Test data loaded. Number of samples: {len(test_dataset)}")

    device = torch.device("cpu")
    logging.info(f"Loading best model from checkpoint...")
    
    image_encoder, img_dim = create_image_encoder(**model_config['image_encoder'])
    model = ImageClassifier(
        image_encoder=image_encoder,
        image_embedding_dim=img_dim,
        mlp_params=model_config['mlp_head']['params']
    )
    
    checkpoint_path = os.path.join(results_dir, 'checkpoints', 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds_proba = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            image = batch['image'].to(device)
            labels = batch['label'] 
            
            outputs = model(batch)
            preds_proba = torch.sigmoid(outputs)
            
            all_labels.append(labels.cpu())
            all_preds_proba.append(preds_proba.cpu())
    
    y_true = torch.cat(all_labels).detach().numpy().flatten()
    y_pred_proba = torch.cat(all_preds_proba).detach().numpy().flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    logging.info("Calculating final metrics on test set...")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_pred_proba)
    }

    logging.info(f"Final Test Metrics: \n{json.dumps(metrics, indent=2)}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    class_names = test_dataset.classes
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(plots_dir, 'test_confusion_matrix.png'))
    plot_roc_curve(y_true, y_pred_proba, os.path.join(plots_dir, 'test_roc_curve.png'))

    logging.info(f"--- Final Evaluation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained image-only model on the test set.")
    parser.add_argument('--model', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data config file.')
    parser.add_argument('--trainer', type=str, required=True, help='Path to the trainer config file.')
    
    args = parser.parse_args()
    evaluate_image_on_test(args.model, args.data, args.trainer)