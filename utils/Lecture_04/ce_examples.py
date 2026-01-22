"""
Civil Engineering examples for optimization demonstrations.

This module provides CE-specific optimization problems and data generation.
"""

import numpy as np


def create_structural_optimization():
    """
    Create a structural beam weight minimization problem.
    
    Returns:
    --------
    dict
        Contains 'objective', 'gradient', 'constraints', 'info'
    
    Problem:
    --------
    Minimize beam weight subject to stress and deflection constraints.
    Design variables: width (w), height (h)
    """
    def objective(x):
        """Beam weight (proportional to cross-sectional area)."""
        w, h = x[0], x[1]
        return w * h  # Simplified: weight ∝ area
    
    def gradient(x):
        """Gradient of objective."""
        return np.array([x[1], x[0]])
    
    def stress_constraint(x):
        """Stress constraint: σ = M*c/I ≤ σ_allowable."""
        w, h = x[0], x[1]
        # Simplified: I = w*h³/12, c = h/2
        # Constraint: stress should be negative (satisfied)
        M = 100  # Bending moment (kN·m)
        sigma_allow = 150  # Allowable stress (MPa)
        I = w * h**3 / 12
        c = h / 2
        sigma = M * c / I if I > 0 else 1e10
        return sigma_allow - sigma  # ≥ 0 for feasibility
    
    def deflection_constraint(x):
        """Deflection constraint: δ ≤ δ_allowable."""
        w, h = x[0], x[1]
        # Simplified: δ = 5*w*L⁴/(384*E*I)
        L = 10  # Span (m)
        E = 200e3  # Young's modulus (MPa)
        w_load = 10  # Distributed load (kN/m)
        I = w * h**3 / 12
        delta_allow = L / 360  # L/360 limit
        delta = 5 * w_load * L**4 / (384 * E * I) if I > 0 else 1e10
        return delta_allow - delta  # ≥ 0 for feasibility
    
    return {
        'objective': objective,
        'gradient': gradient,
        'constraints': [stress_constraint, deflection_constraint],
        'info': {
            'description': 'Beam weight minimization',
            'variables': ['width (m)', 'height (m)'],
            'bounds': [(0.1, 1.0), (0.2, 2.0)]
        }
    }


def create_cost_function():
    """
    Create a CE project cost optimization function.
    
    Returns:
    --------
    dict
        Contains 'cost_fn', 'gradient_fn', 'info'
    
    Problem:
    --------
    Minimize total project cost = material + labor + equipment
    Design variables: concrete strength, steel grade
    """
    def cost_fn(x):
        """
        Total cost function.
        
        Parameters:
        -----------
        x : array [concrete_strength, steel_grade]
        
        Returns:
        --------
        float : Total cost
        """
        fc, fy = x[0], x[1]  # MPa
        
        # Material costs (simplified)
        concrete_cost = 50 + 2 * fc  # $/m³
        steel_cost = 500 + 5 * fy    # $/ton
        
        # Volume/weight requirements (inversely proportional to strength)
        concrete_volume = 1000 / fc  # m³
        steel_weight = 50 / fy       # tons
        
        # Total cost
        total_cost = concrete_cost * concrete_volume + steel_cost * steel_weight
        
        return total_cost
    
    def gradient_fn(x):
        """Numerical gradient of cost function."""
        from .optimizers import compute_gradient
        return compute_gradient(cost_fn, x)
    
    return {
        'cost_fn': cost_fn,
        'gradient_fn': gradient_fn,
        'info': {
            'description': 'Project cost minimization',
            'variables': ['Concrete strength (MPa)', 'Steel grade (MPa)'],
            'typical_range': [(20, 50), (250, 500)]
        }
    }


def create_regression_data(n_samples=100, n_features=1, noise_std=1.0, 
                          true_weights=None, seed=None):
    """
    Generate synthetic regression data for optimization demos.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise_std : float
        Standard deviation of noise
    true_weights : array-like, optional
        True weights (if None, random)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'true_weights', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Add bias column
    X = np.column_stack([np.ones(n_samples), X])
    
    # True weights
    if true_weights is None:
        true_weights = np.random.randn(n_features + 1) * 5
    
    # Generate targets with noise
    y = X @ true_weights + np.random.randn(n_samples) * noise_std
    
    return {
        'X': X,
        'y': y,
        'true_weights': true_weights,
        'info': {
            'n_samples': n_samples,
            'n_features': n_features,
            'noise_std': noise_std
        }
    }


def simulate_overfitting(n_samples=20, degree_range=(1, 15), noise_std=0.5, seed=None):
    """
    Simulate overfitting with polynomial regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples
    degree_range : tuple
        Range of polynomial degrees to test
    noise_std : float
        Noise level
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X_train', 'y_train', 'X_test', 'y_test', 
        'degrees', 'train_errors', 'test_errors'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # True function: y = sin(2πx)
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    # Training data
    X_train = np.sort(np.random.rand(n_samples))
    y_train = true_function(X_train) + np.random.randn(n_samples) * noise_std
    
    # Test data (dense)
    X_test = np.linspace(0, 1, 200)
    y_test = true_function(X_test)
    
    # Test different polynomial degrees
    degrees = range(degree_range[0], degree_range[1] + 1)
    train_errors = []
    test_errors = []
    models = []
    
    for degree in degrees:
        # Create polynomial features
        X_train_poly = np.column_stack([X_train**i for i in range(degree + 1)])
        X_test_poly = np.column_stack([X_test**i for i in range(degree + 1)])
        
        # Fit model (closed-form solution)
        try:
            theta = np.linalg.lstsq(X_train_poly, y_train, rcond=None)[0]
        except:
            theta = np.zeros(degree + 1)
        
        # Predictions
        y_train_pred = X_train_poly @ theta
        y_test_pred = X_test_poly @ theta
        
        # Errors
        train_error = np.mean((y_train - y_train_pred)**2)
        test_error = np.mean((y_test - y_test_pred)**2)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        models.append(theta)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'degrees': list(degrees),
        'train_errors': np.array(train_errors),
        'test_errors': np.array(test_errors),
        'models': models,
        'true_function': true_function
    }


def demonstrate_dropout(n_features=10, dropout_rate=0.5, n_iterations=100, seed=None):
    """
    Demonstrate dropout regularization effect.
    
    Parameters:
    -----------
    n_features : int
        Number of features
    dropout_rate : float
        Probability of dropping a feature (0 to 1)
    n_iterations : int
        Number of dropout iterations
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'weights_history', 'active_features_history', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize weights
    weights = np.random.randn(n_features)
    
    weights_history = [weights.copy()]
    active_features_history = []
    
    for i in range(n_iterations):
        # Apply dropout mask
        mask = np.random.rand(n_features) > dropout_rate
        active_features_history.append(mask.copy())
        
        # Simulate weight update (simplified)
        # In practice, this would be gradient descent with dropout
        gradient = np.random.randn(n_features) * 0.1
        weights = weights - 0.01 * gradient * mask
        
        weights_history.append(weights.copy())
    
    return {
        'weights_history': np.array(weights_history),
        'active_features_history': active_features_history,
        'dropout_rate': dropout_rate,
        'info': {
            'n_features': n_features,
            'n_iterations': n_iterations,
            'avg_active': np.mean([np.sum(mask) for mask in active_features_history])
        }
    }


def generate_classification_data(n_samples=200, n_features=2, n_classes=2, 
                                 separable=True, seed=None):
    """
    Generate synthetic classification data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    separable : bool
        Whether classes are linearly separable
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    
    for i in range(n_classes):
        # Class center
        if separable:
            center = np.random.randn(n_features) * 3
        else:
            center = np.random.randn(n_features) * 0.5
        
        # Generate samples around center
        X_class = np.random.randn(samples_per_class, n_features) + center
        y_class = np.ones(samples_per_class) * i
        
        X.append(X_class)
        y.append(y_class)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    return {
        'X': X,
        'y': y,
        'info': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'separable': separable
        }
    }
