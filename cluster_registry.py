import numpy as np
import time
import numba

from dtw import run_dtwba


def traversal_data(data):

    for index, cluster_dict in enumerate(data):
        print('cluster index ' , index)
        traversal_dict(cluster_dict)
        print()
        
def traversal_dict(my_dict):

    for key, values in my_dict.items():
        print(f"  Cluster {key}: {values}")

def compute_previous(values):
    return ( values + 2 ) / -1

def compute_next(values):
    return -values - 2

def update_records(gp):

    updated_dict = {key: np.where( value < gp.gen_nums, compute_next(value),
            value + gp.data_idx - gp.gen_nums )
        for key, value in gp.cluster_dict.items()
    }
    
    gp.update_reg_hist(updated_dict, gp.hist_idx)
    
    weights = {}
    new_cluster = {}

    for cluster_id, elements in gp.history[-1].items():
        group = []
        weight_ele = []
        for i in elements:

            if i < -1:
                group = np.append(group, gp.result_dict[compute_previous(i)] )
                weight_ele = np.append(weight_ele, np.sum(gp.dtwba_weights[compute_previous(i)]))
            else:
                group = np.append(group, i )
                weight_ele = np.append(weight_ele, 1)
        
        new_cluster[cluster_id] = np.sort(group)
        weights[cluster_id] = weight_ele
        
    gp.result_dict = new_cluster
    gp.dtwba_weights = weights


def compute_label(gp):

    nums = sum(mask.shape[0] for mask in gp.result_dict.values())

    labels = np.full(nums, -999)
    
    for key, mask in gp.result_dict.items():
        labels[mask.astype(int, copy=False)] = int(key)
    
    gp.labels = labels
