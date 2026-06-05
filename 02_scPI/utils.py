from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score


def cluster_scores(latent_space, K, labels_true):
    labels_pred = KMeans(K, n_init=20).fit_predict(latent_space)
    return [
        silhouette_score(latent_space, labels_true),
        NMI(labels_true, labels_pred),
        ARI(labels_true, labels_pred),
    ]


def imputation_error(X_mean, X, X_zero, i, j, ix):
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x - y))
