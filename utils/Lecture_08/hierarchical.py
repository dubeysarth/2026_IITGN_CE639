"""
Hierarchical Clustering

This module provides agglomerative hierarchical clustering with
various linkage methods and dendrogram construction.
"""

import numpy as np


def single_linkage(cluster_a, cluster_b, distance_matrix, cluster_indices):
    """
    Single linkage: minimum distance between any two points.
    
    Parameters:
    -----------
    cluster_a, cluster_b : int
        Cluster indices
    distance_matrix : array_like
        Pairwise distance matrix
    cluster_indices : dict
        Mapping from cluster ID to point indices
    
    Returns:
    --------
    float
        Linkage distance
    """
    min_dist = np.inf
    for i in cluster_indices[cluster_a]:
        for j in cluster_indices[cluster_b]:
            if distance_matrix[i, j] < min_dist:
                min_dist = distance_matrix[i, j]
    return min_dist


def complete_linkage(cluster_a, cluster_b, distance_matrix, cluster_indices):
    """
    Complete linkage: maximum distance between any two points.
    Produces compact, spherical clusters.
    """
    max_dist = 0
    for i in cluster_indices[cluster_a]:
        for j in cluster_indices[cluster_b]:
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
    return max_dist


def average_linkage(cluster_a, cluster_b, distance_matrix, cluster_indices):
    """
    Average linkage: average distance between all pairs.
    """
    total_dist = 0
    count = 0
    for i in cluster_indices[cluster_a]:
        for j in cluster_indices[cluster_b]:
            total_dist += distance_matrix[i, j]
            count += 1
    return total_dist / count if count > 0 else 0


def ward_linkage(cluster_a, cluster_b, X, cluster_indices):
    """
    Ward's linkage: minimizes increase in total WCSS.
    Similar to K-Means objective.
    
    Parameters:
    -----------
    cluster_a, cluster_b : int
        Cluster indices
    X : array_like
        Data points
    cluster_indices : dict
        Mapping from cluster ID to point indices
    
    Returns:
    --------
    float
        Ward distance (increase in WCSS if merged)
    """
    points_a = X[cluster_indices[cluster_a]]
    points_b = X[cluster_indices[cluster_b]]
    
    n_a = len(points_a)
    n_b = len(points_b)
    
    mean_a = np.mean(points_a, axis=0)
    mean_b = np.mean(points_b, axis=0)
    mean_merged = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    
    # Ward distance
    dist = (n_a * n_b / (n_a + n_b)) * np.sum((mean_a - mean_b)**2)
    return np.sqrt(dist)


def compute_linkage_matrix(X, method='average'):
    """
    Compute linkage matrix for hierarchical clustering.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    method : str
        Linkage method: 'single', 'complete', 'average', 'ward'
    
    Returns:
    --------
    array_like, shape (n_samples-1, 4)
        Linkage matrix where each row is [cluster1, cluster2, distance, n_points]
    
    Notes:
    ------
    Compatible with scipy.cluster.hierarchy for plotting.
    """
    n_samples = X.shape[0]
    
    # Compute pairwise distance matrix
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.sqrt(np.sum((X[i] - X[j])**2))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Initialize: each point is its own cluster
    cluster_indices = {i: [i] for i in range(n_samples)}
    active_clusters = set(range(n_samples))
    next_cluster_id = n_samples
    
    linkage_matrix = []
    
    # Select linkage function
    linkage_fn = {
        'single': single_linkage,
        'complete': complete_linkage,
        'average': average_linkage,
    }.get(method, average_linkage)
    
    while len(active_clusters) > 1:
        # Find pair with minimum distance
        min_dist = np.inf
        merge_pair = None
        
        active_list = list(active_clusters)
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                c_a, c_b = active_list[i], active_list[j]
                
                if method == 'ward':
                    dist = ward_linkage(c_a, c_b, X, cluster_indices)
                else:
                    dist = linkage_fn(c_a, c_b, distance_matrix, cluster_indices)
                
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (c_a, c_b)
        
        # Merge clusters
        c_a, c_b = merge_pair
        new_cluster = next_cluster_id
        next_cluster_id += 1
        
        cluster_indices[new_cluster] = cluster_indices[c_a] + cluster_indices[c_b]
        n_points = len(cluster_indices[new_cluster])
        
        linkage_matrix.append([c_a, c_b, min_dist, n_points])
        
        active_clusters.remove(c_a)
        active_clusters.remove(c_b)
        active_clusters.add(new_cluster)
    
    return np.array(linkage_matrix)


def agglomerative_clustering(X, n_clusters=None, distance_threshold=None, method='average'):
    """
    Agglomerative hierarchical clustering.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    n_clusters : int, optional
        Number of clusters (mutually exclusive with distance_threshold)
    distance_threshold : float, optional
        Distance threshold for cutting dendrogram
    method : str
        Linkage method: 'single', 'complete', 'average', 'ward'
    
    Returns:
    --------
    dict
        Contains:
        - 'labels': Cluster assignments
        - 'linkage_matrix': For dendrogram plotting
        - 'n_clusters': Actual number of clusters
    """
    linkage_matrix = compute_linkage_matrix(X, method)
    
    if n_clusters is not None:
        labels = cut_dendrogram(linkage_matrix, n_clusters=n_clusters)
    elif distance_threshold is not None:
        labels = cut_dendrogram(linkage_matrix, distance_threshold=distance_threshold)
    else:
        n_clusters = 2
        labels = cut_dendrogram(linkage_matrix, n_clusters=n_clusters)
    
    return {
        'labels': labels,
        'linkage_matrix': linkage_matrix,
        'n_clusters': len(np.unique(labels))
    }


def cut_dendrogram(linkage_matrix, n_clusters=None, distance_threshold=None):
    """
    Cut dendrogram to obtain cluster labels.
    
    Parameters:
    -----------
    linkage_matrix : array_like, shape (n_samples-1, 4)
        Linkage matrix from compute_linkage_matrix
    n_clusters : int, optional
        Desired number of clusters
    distance_threshold : float, optional
        Maximum linkage distance to consider
    
    Returns:
    --------
    array_like, shape (n_samples,)
        Cluster labels
    """
    n_merges = linkage_matrix.shape[0]
    n_samples = n_merges + 1
    
    # Determine where to cut
    if n_clusters is not None:
        n_merges_to_use = n_samples - n_clusters
    elif distance_threshold is not None:
        n_merges_to_use = 0
        for i in range(n_merges):
            if linkage_matrix[i, 2] <= distance_threshold:
                n_merges_to_use = i + 1
            else:
                break
    else:
        n_merges_to_use = n_merges - 1  # Default: 2 clusters
    
    # Build clusters up to cut point
    cluster_map = {i: i for i in range(n_samples)}
    next_cluster_id = n_samples
    
    for i in range(n_merges_to_use):
        c_a = int(linkage_matrix[i, 0])
        c_b = int(linkage_matrix[i, 1])
        
        # Merge: all points in c_a and c_b now belong to new cluster
        new_id = next_cluster_id
        next_cluster_id += 1
        
        for point, cluster in cluster_map.items():
            if cluster == c_a or cluster == c_b:
                cluster_map[point] = new_id
    
    # Renumber clusters to 0, 1, 2, ...
    labels = np.array([cluster_map[i] for i in range(n_samples)])
    unique_clusters = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_clusters)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels
