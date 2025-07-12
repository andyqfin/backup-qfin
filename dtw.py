import pandas as pd
import numpy as np
from dtaidistance import dtw, dtw_ndim
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from tslearn.barycenters import *

import os
import sys

from numba import njit, jit

from data import *

# set numpy display precision
np.set_printoptions(precision=3, suppress=True, linewidth=200)
np.set_printoptions(threshold=sys.maxsize)


def compute_dtw_i_matrix(data):

    matrix_i = np.zeros( (np.shape(data)[0], np.shape(data)[0]) )
    full_matrix = np.empty( (np.shape(data)[0], 0) )

    for i in range(np.shape(data)[2]):
        xx = data[:, :, i]
        distance_matrix = dtw.distance_matrix_fast(xx, parallel=True)
        full_matrix = np.hstack( (full_matrix, distance_matrix) )
        matrix_i = matrix_i + distance_matrix

    matrix_i = np.array(matrix_i)

    return matrix_i, full_matrix

def perform_dtw_matrix(gp):

    if gp.step == 'step':
        if gp.gen_nums > 0:
            matrix_d = np.array(dtw_ndim.distance_matrix_fast(gp.sequence, block=((0, gp.gen_nums), (gp.gen_nums, np.shape(gp.sequence)[0])),parallel=True))
            matrix_d = matrix_d[gp.gen_nums: np.shape(matrix_d)[0], 0: gp.gen_nums]
            
            matrix = np.array(dtw_ndim.distance_matrix_fast(gp.sequence, block=((0, gp.gen_nums), (0, gp.gen_nums)), parallel=True))
            matrix = matrix[0:gp.gen_nums, 0:gp.gen_nums]
            gp.dtw_matrix = np.vstack((matrix, matrix_d))
        else:

            gp.dtw_matrix = np.array(dtw_ndim.distance_matrix_fast(gp.sequence, parallel=True))

    if gp.step == 'step2':

        gp.sequence = np.vstack((gp.fixed_pattern, gp.pattern, gp.current_data))
        matrix = np.array(dtw_ndim.distance_matrix_fast(gp.sequence, block=( (0, gp.pp2), (gp.pp2, np.shape(gp.sequence)[0])), parallel=True))

        matrix = matrix[gp.pp2: np.shape(gp.sequence)[0], 0: gp.pp2]
        gp.dtw_matrix = matrix

def time_series_average(gp):
    
    gp.pattern = np.vstack([
        gp.sequence[values, :, :] if key == -1 else run_dtwba(gp.sequence[values, :, :], gp.dtwba_weights[key])
        for key, values in gp.cluster_dict.items()
    ])
    

def run_dtwba(data, weights):

    max_iter = 1000
    conv = 1e-6
    int_seq = np.mean(data, axis = 0)

    barycenter = dtw_barycenter_averaging(data, init_barycenter = int_seq, tol = conv, max_iter=max_iter, weights=weights )
    
    return barycenter[np.newaxis, :, :]
