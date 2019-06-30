import numpy as np
import pandas as pd
from math import acos, sin, cos, radians, sqrt
import sys
import time_advance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


geoDist = lambda x, y: acos(sin(radians(x[0]))*sin(radians(y[0]))+cos(radians(x[0]))*cos(radians(y[0]))*cos(radians(x[1]-y[1]))) * 6378.1

def predict_location(fingerprints, latitudes, longitudes, path_loss, bts_pos, current_ta):
    '''Locates an user based on the Path Loss.
    Uses BTS fingerprints and [lat, long] grid.
    '''
    ta_grid = time_advance.get_grid(latitudes, longitudes, bts_pos)
    ta_grid_mask = time_advance.grid_mask(ta_grid, current_ta)
    
    min_distance = sys.float_info.max
    x = -1
    y = -1
    for k in range(latitudes.size):
        for l in range(longitudes.size):
            if(ta_grid_mask[k][l]):
                distance = 0
                for j in range(path_loss.size):
                    distance += pow((fingerprints[j][k][l]-path_loss[j]), 2)
                if(distance < min_distance):
                    min_distance = distance
                    x = k
                    y = l
    return latitudes[x], longitudes[y]

def predict_test_locations(fingerprints, latitudes, longitudes, test, bts_pos):
    '''Locates users based on the BTS fingerprints and [lat, long] grid.
    '''
    errors = []
    y_pred = []
    y = test.iloc[:, 1:3].values
    x = test.iloc[:, 3:9].values
    ids = test.iloc[:, 0:1].values
    tas = test.iloc[:, 9:].values

    for i in range(x.shape[0]):
        lat, long = predict_location(fingerprints, latitudes, longitudes, x[i], bts_pos, tas[i])
        y_pred.append([lat, long])
        errors.append(geoDist((lat, long), (y[i][0], y[i][1]))*1000)
        print('User {:d}/{:d} located.'.format(i, x.shape[0]))

    errors = np.array(errors)
    y_pred = np.array(y_pred)
    lats = [i[0] for i in y]
    lons = [i[1] for i in y]
    lat_preds = [i[0] for i in y_pred]
    lon_preds = [i[1] for i in y_pred]

    results = pd.DataFrame(data={
        'error_loc': errors,
        'lon_pred': lon_preds,
        'lat_pred': lat_preds,
        'lon': lons,
        'lat': lats
    }, index=[i[0] for i in ids])
    results.index.name = "pontoId"
    results = results.sort_index()
    results.to_csv("results/results.csv")

    min_error = np.min(errors)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    std = np.std(errors)
    mse = mean_squared_error(np.zeros(errors.size), errors)
    rmse = sqrt(mse)

    print("\n-- Results --\n")
    print("Min Error: {:.2f}m".format(min_error))
    print("Max Error: {:.2f}m".format(max_error))
    print("Average Error: {:.2f}m".format(mean_error))
    print("Standard Deviation: {:.2f}m".format(std))
    print("Mean Squared Error: {:.2f}m".format(mse))
    print("Root Mean Squared Error (Precision): {:.2f}m".format(rmse))

    fig, ax = plt.subplots()
    ax.hist(errors)
    fig.savefig('results/metrics/error-histogram.png')

    fig, ax = plt.subplots()
    ax.boxplot(errors)
    fig.savefig('results/metrics/error-boxplot.png')

    fig, ax = plt.subplots()
    ax.scatter(lons, lats, color="green", alpha=0.25)
    ax.scatter(lon_preds, lat_preds, color="red", alpha=0.25)
    fig.savefig('results/metrics/location-map.png')
