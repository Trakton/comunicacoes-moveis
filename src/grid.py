import numpy as np
from math import acos, sin, cos, radians
import sys

geoDist = lambda x, y: acos(sin(radians(x[0]))*sin(radians(y[0]))+cos(radians(x[0]))*cos(radians(y[0]))*cos(radians(x[1]-y[1]))) * 6378.1

def point_to_coordinates(point, latitudes, longitudes):
    '''Returns which coordinates in the location grid is
    closer to a [lat, long] point
    '''
    closest = []
    minDist = sys.float_info.max
    for i in range (latitudes.size):
        for j in range (longitudes.size):
            dist = geoDist(point, (latitudes[i], longitudes[j]))
            if dist < minDist:
                minDist = dist
                closest = [i, j]

    return closest[0], closest[1]

def find_bts_coordinates(bts_points, latitudes, longitudes):
    coordinates = []
    for bts in bts_points:
        coordinates.append(point_to_coordinates((bts[0], bts[1]), latitudes, longitudes))
    return coordinates

def build_location_grid(location_points):
    '''Creates a [lat, long] grid from location points.
    Latitudes will go from lower to higher location point. 
    Longitudes will go from leftmost to rightmost.
    '''
    y_min = min(location_points, key=lambda x: x[0])[0]
    y_max = max(location_points, key=lambda x: x[0])[0]
    x_min = min(location_points, key=lambda x: x[1])[1]
    x_max = max(location_points, key=lambda x: x[1])[1]
    
    down_left = (y_min, x_min)
    up_left = (y_max, x_min)
    down_right = (y_min, x_max)
    up_right = (y_max, x_max)

    y_step = max(geoDist(down_left, up_left), geoDist(down_right, up_right))
    x_step = max(geoDist(down_left, down_right), geoDist(up_right, up_left))

    lat_step = ((y_max - y_min) * 1e-2) / y_step
    lon_step = ((x_max - x_min) * 1e-2) / x_step

    latitudes = np.arange(y_min - (lat_step * 10), y_max + (lat_step * 10), lat_step)
    latitudes = (latitudes[:-1] + latitudes[1:]) / 2
    longitudes = np.arange(x_min - (lon_step * 10), x_max + (lon_step * 10), lon_step)
    longitudes = (longitudes[:-1] + longitudes[1:]) / 2

    return latitudes, longitudes