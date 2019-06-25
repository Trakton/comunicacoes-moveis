import numpy as np
from concurrent import futures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from traceback import print_exc

def get_grid(label, model, latitudes, longitudes):
    '''Receives a lat/long grid and computes the path loss
    for each point, using a given model.
    '''
    print("Calculating fingerprint for BTS {:d}".format(label+1))
    pl_grid = np.zeros((latitudes.size, longitudes.size))
    for i in range(latitudes.size):
        for j in range(longitudes.size):
            y_predict = model.predict([[latitudes[i], longitudes[j]]])
            pl_grid[i][j] = y_predict
    print("Fingerprint for BTS {:d} calculated".format(label+1))
    return label, pl_grid

def get_grids_in_parallel(models, latitudes, longitudes):
    '''Receives a lat/long grid and computes the path loss
    for each point, using a given model. Spawn workers to do the job in threads.
    '''
    
    with futures.ProcessPoolExecutor() as executor:
        promises = [executor.submit(get_grid, i, models[i], latitudes, longitudes) for i in range(len(models))]
        futures.wait(promises)

        for promise in promises:
            try:
                yield promise.result()
            except:
                print_exc()

def plot_grid(grid, label, bts_coordinate):
    '''Plots an image of a path loss grid.
    Darker colors mean less path loss, thus, the BTS is closer to that point.
    Lighter colors mean more path loss, thus, the BTS is farther from the point.
    '''
    fig, ax = plt.subplots()
    plt.imshow(grid, cmap='bone', interpolation='nearest', origin='lower')
    ax.add_patch(patches.Rectangle((bts_coordinate[1] - 3,bts_coordinate[0] - 3), 6, 6, hatch='',fill=True, color='r'))
    fig.savefig('results/fingerprints/{:d}.png'.format(label+1))

def get_grids(models, latitudes, longitudes, bts_coordinates):
    '''Receives a lat/long grid and computes the path loss
    for each point, using a given model.
    '''
    grids = []

    for grid in get_grids_in_parallel(models, latitudes, longitudes):
        grids.append(grid)
        plot_grid(grid[1], grid[0], bts_coordinates[grid[0]])

    grids.sort(key=lambda tup: tup[0])
    grids = [i[1] for i in grids]

    print('All fingerprint grids built. Check them at results/fingerprints/ folder.\n')

    return grids