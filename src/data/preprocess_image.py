
import argparse
import yaml
import os
import shutil
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

from src.utils import setup_logging

def preprocess_image_data(data_config_path: str):
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    log_path = os.path.join('results/logs', f"preprocess_{data_config['dataset_name']}.log")
    setup_logging(log_path)

    logging.info(f"--- Starting Image Preprocessing for: {data_config['dataset_name']} ---")
    
    input_dir = data_config['image_root_dir_raw']
    output_dir = data_config['image_root_dir_processed']
    
    val_size = data_config.get('val_size', 0.1)
    test_size = data_config.get('test_size', 0.2)
    seed = data_config.get('seed', 42)
    
    if os.path.exists(output_dir):
        logging.warning(f"Output directory {output_dir} already exists. It will be removed and recreated.")
        shutil.rmtree(output_dir)

    logging.info(f"Creating processed directory structure at: {output_dir}")
    
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not classes:
        logging.error(f"No class subdirectories found in {input_dir}. Exiting.")
        return
    logging.info(f"Found classes: {classes}")

    for cls in classes:
        logging.info(f"\nProcessing class: {cls}")
        
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)
        
        class_path = os.path.join(input_dir, cls)
        all_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        if not all_files:
            logging.warning(f"No files found in class directory: {class_path}")
            continue

        train_val_files, test_files = train_test_split(all_files, test_size=test_size, random_state=seed)
        
        val_size_adjusted = val_size / (1 - test_size)
        train_files, val_files = train_test_split(train_val_files, test_size=val_size_adjusted, random_state=seed)
        
        logging.info(f"  Split sizes: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        for f in tqdm(train_files, desc=f"Copying {cls} train files"):
            shutil.copy(os.path.join(class_path, f), os.path.join(output_dir, 'train', cls, f))
        for f in tqdm(val_files, desc=f"Copying {cls} val files"):
            shutil.copy(os.path.join(class_path, f), os.path.join(output_dir, 'val', cls, f))
        for f in tqdm(test_files, desc=f"Copying {cls} test files"):
            shutil.copy(os.path.join(class_path, f), os.path.join(output_dir, 'test', cls, f))
            
    logging.info("\n--- Image Preprocessing Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split image folder into train/val/test and copy to processed folder.")
    parser.add_argument('--config', type=str, required=True, help='Path to the data config file for image data.')
    args = parser.parse_args()
    preprocess_image_data(args.config)