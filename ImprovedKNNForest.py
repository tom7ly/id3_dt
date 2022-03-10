from CostSensetiveID3 import P_params
from operator import sub
import matplotlib
from decicsion_tree import Tree, Classifier
import numpy as np
import sklearn
import pandas as pd
import matplotlib
from math import floor, sqrt
from typing import List
from sklearn import model_selection
import time
from numpy.random import default_rng as rng
from knn_forest import *

if __name__ == "__main__":

    'To run the improved KNN KFold experiemnt, uncomment the following function call'
    # KNN_InitKFoldForest( N_params=[6, 10, 12, 14, 16, 20, 32, 64, 128],
    #     P_params=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    #     K_params=[4, 4, 6, 14, 16, 20, 32, 64, 128],modified=True)

    'To run the improved KNN Forest, call the function with parameter modified = True'
    print(KNN_InitKNNForest(N_forest_size=60, K_closest=20, P_param=0.45, modified=True))
