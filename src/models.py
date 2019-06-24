def train(X, Y, model):
    '''Trains a model for each column in the Y dataset
    '''
    models = []

    for i in range(Y.shape[1]):
        print("Training model for BTS%d" % (i+1))
        x_predict = X[:, :]
        y_predict = Y[:, i:(i+1)]
        trained_model = model.fit(X, Y)
        models.append(trained_model)

    return models