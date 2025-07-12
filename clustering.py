import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics.cluster import adjusted_rand_score
import time
import matplotlib.pyplot as plt

from cluster_registry import traversal_dict

from sklearn.cluster import KMeans

from tqdm.auto import tqdm

# import cuml

def get_kneedle(k_dist):

    knee_y_value = KneeLocator(x = range(1, np.shape(k_dist)[0] + 1 ), y = k_dist, S = 1, curve = 'convex', direction = "increasing", online=True).knee_y

    if knee_y_value == 0:
        knee_y_value = np.max(k_dist)

    return knee_y_value

def get_kdist(data):

    return np.sort(NearestNeighbors(n_neighbors=2).fit(data).kneighbors(data)[0][:, -1])

def get_cluster(labels):

    return {label: np.where(labels == label)[0] for label in np.unique(labels)}

def run_dbscan(gp):

    data = StandardScaler().fit_transform(gp.dtw_matrix)
    
    gp.n_values = min(max(math.ceil(gp.fixed_n_values / gp.hist_idx), gp.n_values), math.ceil(np.shape(data)[1] / 2))
    
    while (True):
        ipca = IncrementalPCA(n_components=gp.n_values, batch_size=5 * gp.n_values)
        X_pca = ipca.fit_transform(data)
        
        if (np.sum(ipca.explained_variance_ratio_) < gp.ratio):
            gp.n_values = gp.n_values + 1
        else:
            break
    
    gp.fixed_n_values = gp.fixed_n_values + gp.n_values
    
    kneedle = get_kneedle(get_kdist(X_pca))
    
    result = DBSCAN(eps=kneedle, min_samples=1, n_jobs=-1).fit(X_pca)
    
    gp.labels = result.labels_
    gp.cluster_dict = get_cluster(result.labels_)

def get_actual_nums(entry):

    noise_counter = 0
    counter = 0

    for key, values in entry.get("grouped_clusters").items():
        if key == -1:
            noise_counter = np.shape(values)[0]
            counter = counter + noise_counter
        if key != -1:
            counter = counter + 1
            
    return counter, noise_counter

def clustering_score(a, b, evaluate):

    if evaluate.lower() == "ari".lower():
        score = adjusted_rand_score(a, b)
        
    print(score)
    
    return score


def latent_dbscan(latent):

    latent = StandardScaler().fit_transform(latent)

    print('run HDBSCAN')
    result = HDBSCAN(min_samples=1, n_jobs=4).fit(latent)

    labels = result.labels_
    cluster_dict = get_cluster(result.labels_)

    # traversal_dict((cluster_dict))

    for key, values in cluster_dict.items():
        print(f"  Cluster {key}: {np.shape(values)}")

    return result

def elbow_method(data):

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    scores_1 = []

    print(np.shape(data)[0])

    for i in tqdm( range(1, np.shape(data)[0], 100), desc="Processing"):
        kmeans = MiniBatchKMeans(n_clusters=i, batch_size=1024)
        kmeans.fit(data)
        scores_1.append(kmeans.inertia_)
    print(scores_1)

    yy = get_kneedle(scores_1)

    print(yy)

    for i in tqdm( range(1, np.shape(data)[0], 100), desc="Processing"):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        scores_1.append(kmeans.inertia_)
    print(scores_1)


def run_dbscan2(data):

    data = StandardScaler().fit_transform(data)
    kneedle = get_kneedle(get_kdist(data))

    result = DBSCAN(eps=kneedle, min_samples=2, n_jobs=4).fit(data)

    labels = result.labels_
    cluster_dict = get_cluster(result.labels_)

    traversal_dict(cluster_dict)

def get_xmeans(X):

    xm = XMeans(X)
    xm.fit()

    print("Estimated k = " + str(xm.k) )

def compare_kmeans_time(data, num):

    s = time.process_time()  # start time

    batch_size = 256 * num
    kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=batch_size)
    kmeans.fit(data)

    e = time.process_time()  # end time
    ss = e-s

    aa = time.process_time()  # start time

    batch_size = 256 * num
    kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=batch_size)
    kmeans.fit(data)

    bb = time.process_time()  # end time
    sss = bb-aa
    print(num, batch_size, ss, "seconds", sss)
    return ss