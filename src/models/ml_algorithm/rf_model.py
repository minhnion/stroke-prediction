from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'class_weight': 'balanced', 
                'n_jobs': -1 
            }
        else:
            self.params = params
            
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)