from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class MLPModel:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse
    