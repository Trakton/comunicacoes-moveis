from copy import deepcopy
import numpy as np

def train(X, Y, model):
    '''Trains a model for each column in the Y dataset
    '''
    models = []

    for i in range(Y.shape[1]):
        print("Training model for BTS%d" % (i+1))
        x_predict = X[:, :]
        y_predict = Y[:, i:(i+1)]
        trained_model = deepcopy(model)
        trained_model.fit(x_predict, y_predict)
        models.append(trained_model)

    return models