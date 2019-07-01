import numpy as np
from sklearn import base as skBase
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ErrorModel(skBase.BaseEstimator):

    def __init__(self, predictor, error_predictor):
        self.predictor = predictor
        self.error_predictor = error_predictor
        self.x_scaler = preprocessing.RobustScaler()
        self.y_scaler = preprocessing.RobustScaler()

    def fit(self, x, y):
        half = int(x.shape[0] / 2)
        x_scaled = self.x_scaler.fit_transform(np.array(x[:half]))
        y_scaled = self.y_scaler.fit_transform(y[:half]).ravel()
        x_error_scaled = self.x_scaler.transform(np.array(x[half:]))
        y_error_scaled = self.y_scaler.transform(y[half:]).ravel()
        self.predictor = self.predictor.fit(x_scaled, y_scaled)
        prediction_errors = y_error_scaled - self.predictor.predict(x_error_scaled)
        self.error_predictor = self.error_predictor.fit(x_error_scaled, prediction_errors)
    
    def predict(self, x):
        x_scaled = self.x_scaler.transform(x)
        prediction = self.predictor.predict(x_scaled)
        pred = [prediction]
        error = self.error_predictor.predict(x_scaled)
        prediction = prediction + error
        return self.y_scaler.inverse_transform(pred)