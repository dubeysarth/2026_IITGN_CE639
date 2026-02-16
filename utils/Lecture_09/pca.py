"""
Principal Component Analysis (PCA)

This module implements PCA from scratch using both:
1. Covariance matrix + eigendecomposition (pedagogical)
2. SVD (numerically stable, production-ready)
"""

import numpy as np


def standardize_data(X):
    """
    Standardize data using Z-score normalization.
    
    Z = (X - μ) / σ
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data matrix
    
    Returns:
    --------
    dict
        Contains:
        - 'Z': Standardized data
        - 'mean': Feature means
        - 'std': Feature standard deviations
    
    Notes:
    ------
    PCA requires standardization to ensure all features
    contribute equally to variance calculation.
    """
    X = np.asarray(X, dtype=float)
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)  # Sample std (n-1)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    Z = (X - mean) / std
    
    return {
        'Z': Z,
        'mean': mean,
        'std': std
    }


def pca_covariance(X, n_components=None):
    """
    PCA via covariance matrix and eigendecomposition.
    
    Algorithm:
    1. Standardize: Z = (X - μ) / σ
    2. Covariance: C = (1/(n-1)) * Z^T Z
    3. Eigendecomposition: C = V Λ V^T
    4. Sort by eigenvalues (descending)
    5. Select top k eigenvectors
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data matrix
    n_components : int, optional
        Number of components to keep. If None, keep all.
    
    Returns:
    --------
    dict
        Contains:
        - 'components': Principal components (eigenvectors), shape (n_components, n_features)
        - 'explained_variance': Variance explained by each PC
        - 'explained_variance_ratio': Fraction of variance explained
        - 'mean': Feature means
        - 'std': Feature stds
        - 'Z': Standardized data
    
    Notes:
    ------
    This method is pedagogically clear but can be numerically
    unstable for large feature spaces. Use pca_svd() for production.
    """
    n_samples, n_features = X.shape
    
    # Standardize
    std_result = standardize_data(X)
    Z = std_result['Z']
    
    # Covariance matrix: C = (1/(n-1)) * Z^T Z
    C = (1 / (n_samples - 1)) * (Z.T @ Z)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select components
    if n_components is None:
        n_components = n_features
    
    components = eigenvectors[:, :n_components].T  # Shape: (n_components, n_features)
    explained_variance = eigenvalues[:n_components]
    
    # Variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_variance
    
    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'mean': std_result['mean'],
        'std': std_result['std'],
        'Z': Z,
        'n_components': n_components
    }


def pca_svd(X, n_components=None):
    """
    PCA via Singular Value Decomposition (SVD).
    
    Algorithm:
    1. Standardize: Z = (X - μ) / σ
    2. SVD: Z = U Σ V^T
    3. Principal components = V (right singular vectors)
    4. Eigenvalues = σ²/(n-1)
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data matrix
    n_components : int, optional
        Number of components to keep
    
    Returns:
    --------
    dict
        Same as pca_covariance()
    
    Notes:
    ------
    SVD is numerically stable and avoids computing Z^T Z explicitly.
    This is the preferred method for production code.
    
    Mathematical equivalence:
    Z^T Z = (U Σ V^T)^T (U Σ V^T) = V Σ^T U^T U Σ V^T = V Σ^2 V^T
    
    So V contains the eigenvectors of the covariance matrix.
    """
    n_samples, n_features = X.shape
    
    # Standardize
    std_result = standardize_data(X)
    Z = std_result['Z']
    
    # SVD: Z = U Σ V^T
    U, singular_values, Vt = np.linalg.svd(Z, full_matrices=False)
    
    # V is the transpose of Vt
    V = Vt.T
    
    # Eigenvalues from singular values
    eigenvalues = (singular_values ** 2) / (n_samples - 1)
    
    # Select components
    if n_components is None:
        n_components = min(n_samples, n_features)
    
    components = V[:, :n_components].T  # Shape: (n_components, n_features)
    explained_variance = eigenvalues[:n_components]
    
    # Variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_variance
    
    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'mean': std_result['mean'],
        'std': std_result['std'],
        'Z': Z,
        'n_components': n_components,
        'singular_values': singular_values[:n_components]
    }


def explained_variance_ratio(pca_result):
    """
    Compute cumulative explained variance ratio.
    
    Parameters:
    -----------
    pca_result : dict
        Result from pca_covariance() or pca_svd()
    
    Returns:
    --------
    dict
        Contains:
        - 'individual': Variance ratio per component
        - 'cumulative': Cumulative variance ratio
    """
    individual = pca_result['explained_variance_ratio']
    cumulative = np.cumsum(individual)
    
    return {
        'individual': individual,
        'cumulative': cumulative
    }


def transform_pca(X, pca_result):
    """
    Project data onto principal components.
    
    Z_PC = Z * V_k
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data to transform
    pca_result : dict
        Result from pca_covariance() or pca_svd()
    
    Returns:
    --------
    array_like, shape (n_samples, n_components)
        Transformed data in PC space
    """
    # Standardize using same mean/std
    Z = (X - pca_result['mean']) / pca_result['std']
    
    # Project: Z_PC = Z @ V_k^T
    Z_PC = Z @ pca_result['components'].T
    
    return Z_PC


def inverse_transform_pca(Z_PC, pca_result):
    """
    Reconstruct data from PC space.
    
    Z_reconstructed = Z_PC * V_k^T
    X_reconstructed = Z_reconstructed * σ + μ
    
    Parameters:
    -----------
    Z_PC : array_like, shape (n_samples, n_components)
        Data in PC space
    pca_result : dict
        Result from pca_covariance() or pca_svd()
    
    Returns:
    --------
    array_like, shape (n_samples, n_features)
        Reconstructed data in original space
    """
    # Reconstruct standardized data
    Z_reconstructed = Z_PC @ pca_result['components']
    
    # Unstandardize
    X_reconstructed = Z_reconstructed * pca_result['std'] + pca_result['mean']
    
    return X_reconstructed


def pca_step_by_step(X, n_components=2, method='svd'):
    """
    Generator that yields PCA computation steps for visualization.
    
    Parameters:
    -----------
    X : array_like
        Data matrix
    n_components : int
        Number of components
    method : str
        'covariance' or 'svd'
    
    Yields:
    -------
    dict
        State at each step with 'step', 'description', 'data'
    """
    n_samples, n_features = X.shape
    
    # Step 1: Original data
    yield {
        'step': 1,
        'description': 'Original Data',
        'data': X.copy(),
        'type': 'original'
    }
    
    # Step 2: Standardization
    std_result = standardize_data(X)
    Z = std_result['Z']
    
    yield {
        'step': 2,
        'description': 'Standardized Data (Z-score)',
        'data': Z.copy(),
        'mean': std_result['mean'],
        'std': std_result['std'],
        'type': 'standardized'
    }
    
    if method == 'covariance':
        # Step 3: Covariance matrix
        C = (1 / (n_samples - 1)) * (Z.T @ Z)
        
        yield {
            'step': 3,
            'description': 'Covariance Matrix',
            'data': C.copy(),
            'type': 'covariance'
        }
        
        # Step 4: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        yield {
            'step': 4,
            'description': 'Eigendecomposition',
            'eigenvalues': eigenvalues.copy(),
            'eigenvectors': eigenvectors.copy(),
            'type': 'eigen'
        }
        
    else:  # SVD
        # Step 3: SVD
        U, singular_values, Vt = np.linalg.svd(Z, full_matrices=False)
        eigenvalues = (singular_values ** 2) / (n_samples - 1)
        
        yield {
            'step': 3,
            'description': 'Singular Value Decomposition',
            'singular_values': singular_values.copy(),
            'eigenvalues': eigenvalues.copy(),
            'V': Vt.T.copy(),
            'type': 'svd'
        }
    
    # Step 5: Projection
    if method == 'covariance':
        components = eigenvectors[:, :n_components].T
    else:
        components = Vt[:n_components]
    
    Z_PC = Z @ components.T
    
    yield {
        'step': 4 if method == 'svd' else 5,
        'description': f'Projection onto {n_components} PCs',
        'data': Z_PC.copy(),
        'components': components.copy(),
        'type': 'projection'
    }
    
    # Step 6: Reconstruction
    Z_reconstructed = Z_PC @ components
    X_reconstructed = Z_reconstructed * std_result['std'] + std_result['mean']
    
    yield {
        'step': 5 if method == 'svd' else 6,
        'description': 'Reconstruction',
        'data': X_reconstructed.copy(),
        'reconstruction_error': np.mean((X - X_reconstructed) ** 2),
        'type': 'reconstruction'
    }
