import numpy as np
from math import acos, sin, cos, radians
import sys
import time_advance

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
    y = test.iloc[:, 1:3].values
    x = test.iloc[:, 3:9].values
    tas = test.iloc[:, 9:].values

    for i in range(x.shape[0]):
        lat, long = predict_location(fingerprints, latitudes, longitudes, x[i], bts_pos, tas[i])
        error = geoDist((lat, long), (y[i][0], y[i][1]))
        errors.append(error)
        print('User {:d} located at {:.6f}, {:.6f}.'.format(i, lat, long))

    errors = np.array(errors)
    std = np.std(errors)
    mse = np.mean((errors**2))
    print('MSE: {:.3f} STD: {:.3f}'.format(mse, std))
