import numpy as np
from sklearn import base as skBase
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Scaler(skBase.BaseEstimator):

    def __init__(self, model):
        self.model = model    
        self.x_scaler = preprocessing.RobustScaler()
        self.y_scaler = preprocessing.RobustScaler()

    def fit(self, x, y):
        x_scaled = y_scaled = []

        x_scaled = self.x_scaler.fit_transform(np.array(x))
        y_scaled = self.y_scaler.fit_transform(y).ravel()

        self.model = self.model.fit(x_scaled, y_scaled)

    def predict(self, x):
        x_scaled = self.x_scaler.transform(x)
        prediction = self.model.predict(x_scaled)
        
        return self.y_scaler.inverse_transform(prediction)
