from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'random_seed': 42,
                'verbose': 0,
                'thread_count': -1
            }
        else:
            self.params = params
            
        self.model = CatBoostClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=None):
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds
        )
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)