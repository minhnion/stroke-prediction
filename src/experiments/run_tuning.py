import argparse
import yaml
import os
import pandas as pd
import json
import logging
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from tqdm import tqdm 

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

from src.utils import setup_logging, tqdm_joblib 

def run_tuning(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(config['data_config_path'], 'r') as f:
        data_config = yaml.safe_load(f)
        
    exp_name = config['experiment_name']
    
    results_dir = os.path.join('results/experiments', exp_name)
    best_params_path = os.path.join(results_dir, 'best_params.json')
    tuning_results_path = os.path.join(results_dir, 'tuning_results.csv')
    os.makedirs(results_dir, exist_ok=True)
    
    log_path = os.path.join('results/logs', f"{exp_name}.log")
    setup_logging(log_path)
    
    logging.info(f"--- Starting Hyperparameter Tuning for: {config['model']['name']} ---")
    
    processed_dir = data_config['processed_path']
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv')).values.ravel()
    
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([pd.Series(y_train), pd.Series(y_val)], ignore_index=True)
    logging.info(f"Combined data for tuning: {X.shape}")
    
    model_name = config['model']['name']
    
    if model_name == "SVMModel":
        logging.info("Applying One-Hot Encoding for SVM...")
        categorical_cols = data_config['categorical_features']
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
        logging.info(f"Data shape after One-Hot Encoding: {X.shape}")
    
    models = {
        "XGBoostModel": XGBClassifier(eval_metric='logloss'),
        "RandomForestModel": RandomForestClassifier(),
        "LightGBMModel": LGBMClassifier(),
        "CatBoostModel": CatBoostClassifier(verbose=0),
        "SVMModel": SVC()
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model name for tuning: {model_name}")
    
    estimator = models[model_name]
    
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
        if model_name in ["XGBoostModel", "LightGBMModel", "CatBoostModel"]:
            estimator.set_params(scale_pos_weight=scale_pos_weight)
        elif model_name in ["RandomForestModel", "SVMModel"]:
            estimator.set_params(class_weight='balanced')
    
    param_dist = config['model']['param_dist']
    tuning_cfg = config['tuning']
    
    cv_strategy = StratifiedKFold(n_splits=tuning_cfg['cv'], shuffle=True, random_state=42)
    scorer = make_scorer(f1_score) if tuning_cfg['scoring'] == 'f1' else tuning_cfg['scoring']

    logging.info(f"Starting RandomizedSearchCV with n_iter={tuning_cfg['n_iter']} and cv={tuning_cfg['cv']}...")
    
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=tuning_cfg['n_iter'],
        scoring=scorer,
        n_jobs=tuning_cfg['n_jobs'],
        cv=cv_strategy,
        random_state=42,
        verbose=0 
    )
    

    total_fits = tuning_cfg['n_iter'] * tuning_cfg['cv']
    
    with tqdm(total=total_fits, desc="Tuning Progress") as pbar:
        with tqdm_joblib(pbar):
            random_search.fit(X, y)
    
    logging.info("Tuning finished.")
    logging.info(f"Best score ({tuning_cfg['scoring']}): {random_search.best_score_:.4f}")
    
    best_params = random_search.best_params_
    logging.info(f"Best parameters found: \n{json.dumps(best_params, indent=2)}")
        
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"Best parameters saved to {best_params_path}")
    
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df.to_csv(tuning_results_path, index=False)
    logging.info(f"Full tuning results saved to {tuning_results_path}")

    logging.info(f"--- Hyperparameter Tuning Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning from a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the tuning config YAML file.')
    args = parser.parse_args()
    run_tuning(args.config)