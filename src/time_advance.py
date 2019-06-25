import numpy as np
from math import acos, sin, cos, radians, floor

geoDist = lambda x, y: acos(sin(radians(x[0]))*sin(radians(y[0]))+cos(radians(x[0]))*cos(radians(y[0]))*cos(radians(x[1]-y[1]))) * 6378.1

def time_advance(x, y):
    return floor( geoDist(x,y)/0.55)

def grid_mask(ta_grid, mobile_ta):
    mask = np.zeros((len(ta_grid), len(ta_grid[0])), dtype=bool) 
    for match_idx in range(len(mobile_ta),1,-1):
        for i in range(len(ta_grid)):
            for j in range(len(ta_grid[0])):
                count = 0
                for k in range(len(mobile_ta)):
                    if(ta_grid[i][j][k] == mobile_ta[k]):
                        count += 1
                if(count == match_idx):
                    mask[i][j] = True    
        if(mask.any()): 
            return mask
    return mask

def get_grid(latitudes, longitudes, erbs_pos):
    ta_matrix = []
    for i in range(latitudes.size):
        ta_matrix.append([])
        for j in range(longitudes.size):
            ta_matrix[i].append([])
            for k in range(len(erbs_pos)):
                x = (latitudes[i], longitudes[j])
                y = (erbs_pos[k][0], erbs_pos[k][1])
                ta_matrix[i][j].append(time_advance(x, y))
    return ta_matrix