import pandas as pd
import numpy as np
import grid
import models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def main():
    train_data = pd.read_csv('data/train.csv')
    bst_data = pd.read_csv('data/bts.csv')

    train_data = shuffle(train_data)
    train, test = train_test_split(train_data, test_size=0.1)
    train = train.dropna()
    test = test.dropna()

    train_points = train.iloc[:, 1:3].values
    train_path_loss = train.iloc[:, 3:9].values
    bst_points = bst_data.iloc[:, 1:3].values
    location_points = np.concatenate((train_points, bst_points), axis=0)
    
    latitudes, longitudes = grid.build_location_grid(location_points)
    print("location grid with size [{:d}, {:d}] calculated.".format(latitudes.shape[0], longitudes.shape[0]))

    knn = KNeighborsRegressor(n_neighbors=5)
    trained_models = models.train(train_points, train_path_loss, knn)
    #f = open('results/knn.txt', 'w')
    

if __name__ == '__main__':
    main()