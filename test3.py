from typing import Any

import numpy as np
from numpy import ndarray, dtype
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def compute_mutual_reachability(data, min_samples=2):
    k = min_samples   # min_samples=2时k=1
    data = np.sort(data.ravel()).reshape(-1, 1)  # 强制排序
    
    
    min_samples = 2
    
    # Compute the k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
   
    core_distances = distances[:, -1]
    
    # Print results
    for i, core_distance in enumerate(core_distances):
        print(f"Point {data[i]} -> Core Distance: {core_distance:.4f}")
    
    
    
    # 计算一维绝对值距离矩阵
    dist_mat = np.abs(data - data.T)
    print('sssssssss')
    print(dist_mat)
    # 计算核心距离
    core_distances = np.zeros(len(data))
    for i in range(len(data)):
        sorted_dists = np.sort(dist_mat[i])[1:]  # 排除自身
        core_distances[i] = sorted_dists[k]
    print(core_distances)
    # 计算相互可达距离
    mreach_dist = np.zeros_like(dist_mat)
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                mreach_dist[i, j] = max(core_distances[i], core_distances[j], dist_mat[i, j])
    
    return mreach_dist


# 测试数据（用户提供的样本）
data: ndarray[Any, dtype[Any]] = np.array([-0.672, -0.353, 0.045, 7.543, 9.181])
min_samples = 2  # 对应k=2

mreach_matrix = compute_mutual_reachability(data, min_samples)

print(mreach_matrix)

