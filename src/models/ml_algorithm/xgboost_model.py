import xgboost as xgb
import pandas as pd

class XGBoostModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 4,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
            }
        else:
            self.params = params
        
        self.model = xgb.XGBClassifier(**self.params)
    
    def fit(self, X_train, y_train, X_val, y_val, early_stopping_rounds=50):
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        print(f"Using scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
        
        self.model.set_params(
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds 
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False  
        )
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)