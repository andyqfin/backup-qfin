# Environment configuration (must be set before TF import)
import os
import sys

from sklearn.cluster import DBSCAN
from sklearn.decomposition import IncrementalPCA

from clustering import get_cluster, get_kneedle, get_kdist, clustering_score

from cluster_registry import *
from dtw import *
# from neural import neural_optimizer
from progress_bar import show_progress

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF info logs

# Core scientific computing
import numpy as np
import pandas as pd
import pickle
import math
# Visualization setup
# import matplotlib.pyplot as plt

# set numpy display precision
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

class GiantsParams:
    def __init__(self, data, batch_size):
        
        self.step = None
        self.current_data = None
        
        self.sequence = None
        self.dtw_matrix = None
        
        self.dep = True
        
        self.ratio = 0.999
        
        self.fixed_n_values = 10
        self.n_values = 10
        self.hist_idx = 0
        self.data_idx = 0
        self.next = 0
        
        self.batch_size = batch_size
        self.save = 'save' + str(batch_size)
        self.data = data
        self.result_dict = None
        self.labels = None
        
        self.history = []
        
        self.pattern = np.empty((0, np.shape(data)[1], np.shape(data)[2]))
        self.gen_nums = 0
        
        self.cluster_dict = None
        self.dtwba_weights = None
    
        self.fixed_pattern = None
    
        self.pp2 = None
    
    def update_reg_hist(self, dicts, index):
    
        while len(self.history) <= index - 1:
            self.history.append({})
        
        self.history[index - 1].update(dicts)
        
    def gp_update(self, data):
    
        self.data = np.vstack( (self.data, data) )
        
def giants(gp):
    
    gp.step = 'step'
    
    giants_process(gp)
    
    next_gp = GiantsParams(gp.data, batch_size=gp.batch_size)
    next_gp.step = 'step2'
    next_gp.fixed_pattern = gp.pattern

    next_gp.pp2 = np.shape(next_gp.fixed_pattern)[0]

    giants_process(next_gp)
    
    clustering_score(next_gp.labels, gp.labels, 'ari')
    
    return next_gp
    
    # pickle.dump(gp, open(gp.save, "wb"))

def giants_process(gp):

    batches, progress = show_progress(gp.batch_size, np.shape(gp.data)[0] - gp.data_idx)

    while gp.data_idx < np.shape(gp.data)[0]:
        gp.hist_idx = gp.hist_idx + 1
        gp.next = min(gp.data_idx + gp.batch_size, np.shape(gp.data)[0])
        gen_pattern(gp)
        gp.data_idx = gp.next
        progress.update()
    
    progress.close()
    compute_label(gp)
    
    # traversal_dict(gp.result_dict)
    

def gen_pattern(gp):

    gp.gen_nums = np.shape(gp.pattern)[0]
    gp.current_data = gp.data[gp.data_idx: gp.next, :, :]

    feature_selection(gp)
    update_records(gp)

    time_series_average(gp)

def feature_selection(gp):

    gp.sequence = np.vstack((gp.pattern, gp.current_data))
    
    perform_dtw_matrix(gp)
    
    run_dbscan(gp)

def gen_pattern2(gp):

    gp.current_data = gp.data[gp.data_idx: gp.next, :, :]
    feature_selection(gp)
    update_records(gp)
    time_series_average(gp)