from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def blob_demo():
    '''
    Create and visualize an artificial data set for basic classification tasks.
    There are three features, and the third feature has no actual importance.
    '''
    ###################
    ## DATA SET PREP ##
    ###################
    X, y = make_blobs(
        n_samples = 5000, 
        n_features = 3,
        centers = [
            [0,1,-1], [1,0,-1], [0,1,1], [1,0,1],  # will be negative cases
            [0,0,-1], [1,1,-1], [0,0,1], [1,1,1]   # will be positive cases
        ],
        cluster_std = .5,  # spread the "blobs" out enough for slight difficulty
        shuffle = True
    )
    # Put the features in pandas DataFrame
    X = pd.DataFrame(X, columns = ['feature1', 'feature2', 'feature3'])
    # Convert the original labels from [0,1,2,3,4,5,6,7] to just [0,1]
    y = (y >= 4).astype(int)  
    # Convert to pandas 
    y = pd.Series(y)

    ###################
    ## VISUALIZATION ##
    ###################   
    plt.rcParams['figure.figsize'] = 24,8    # Plot size and
    plt.rcParams['font.size'] = 14           # font size

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    # Show the data set from "above" (i.e. in 2D form)
    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(90, 270)
    ax.scatter(
        X.feature1, X.feature2, X.feature3, 
        c = ['b' if pt == 1 else 'r' for pt in y],    # blue = pos, red = neg
        s = 2
    )
    ax.set_xlabel('feature1')
    ax.set_ylabel('feature2')
    ax.set_zlabel('feature3')
    ax.set_xlim(-2,3)
    ax.set_ylim(-2,3)
    ax.set_zlim(-3,3)

    # Show again in 3D from the default angle
    ax = fig.add_subplot(132, projection='3d')
    ax.view_init()
    ax.scatter(
        X.feature1, X.feature2, X.feature3, 
        c = ['b' if pt == 1 else 'r' for pt in y], 
        s = 2
    )
    ax.set_xlabel('feature1')
    ax.set_ylabel('feature2')
    ax.set_zlabel('feature3')
    ax.set_xlim(-2,3)
    ax.set_ylim(-2,3)
    ax.set_zlim(-3,3)

    # Show again in 3D from another angle
    ax = fig.add_subplot(133, projection='3d')
    ax.view_init(30, 210)
    ax.scatter(
        X.feature1, X.feature2, X.feature3, 
        c = ['b' if pt == 1 else 'r' for pt in y], 
        s = 2#alpha = .2
    )
    ax.set_xlabel('feature1')
    ax.set_ylabel('feature2')
    ax.set_zlabel('feature3')
    ax.set_xlim(-2,3)
    ax.set_ylim(-2,3)
    ax.set_zlim(-3,3)

    plt.show()
    return X, y
