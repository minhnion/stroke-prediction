from sklearn.svm import SVC

class SVMModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,  
                'class_weight': 'balanced', 
                'random_state': 42
            }
        else:
            self.params = params
            
        self.model = SVC(**self.params)

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)