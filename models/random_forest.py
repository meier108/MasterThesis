import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    