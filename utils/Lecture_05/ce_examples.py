"""
Civil Engineering examples for vector calculus applications.

This module provides CE-specific examples demonstrating gradients
and sensitivity analysis.
"""

import numpy as np


def structural_stress_gradient():
    """
    Compute stress gradient for sensitivity analysis.
    
    Example: Beam stress σ = M*c/I where I = b*h³/12
    
    Returns:
    --------
    dict
        Contains stress function, gradient function, and example
    """
    def stress(params):
        """
        Compute beam stress.
        
        Parameters:
        -----------
        params : array [b, h]
            b: beam width (m)
            h: beam height (m)
        
        Returns:
        --------
        float
            Stress (MPa)
        """
        b, h = params[0], params[1]
        M = 100  # Bending moment (kN·m)
        c = h / 2  # Distance to neutral axis
        I = b * h**3 / 12  # Moment of inertia
        
        if I == 0:
            return 1e10
        
        sigma = M * c / I
        return sigma
    
    def stress_gradient(params):
        """
        Analytical gradient of stress.
        
        Returns:
        --------
        array
            [∂σ/∂b, ∂σ/∂h]
        """
        b, h = params[0], params[1]
        M = 100
        
        # ∂σ/∂b = -M*h/(2*b²*h³/12) = -6M/(b²*h²)
        dsigma_db = -6 * M / (b**2 * h**2)
        
        # ∂σ/∂h = -M/(b*h²)
        dsigma_dh = -M / (b * h**2)
        
        return np.array([dsigma_db, dsigma_dh])
    
    # Example calculation
    params = np.array([0.3, 0.5])  # b=0.3m, h=0.5m
    sigma_val = stress(params)
    grad_val = stress_gradient(params)
    
    return {
        'stress_fn': stress,
        'gradient_fn': stress_gradient,
        'example': {
            'params': params,
            'stress': sigma_val,
            'gradient': grad_val,
            'interpretation': {
                'dsigma_db': f'{grad_val[0]:.2f} MPa/m (stress decreases as width increases)',
                'dsigma_dh': f'{grad_val[1]:.2f} MPa/m (stress decreases as height increases)'
            }
        }
    }


def deflection_sensitivity():
    """
    Compute deflection sensitivity for beam design.
    
    Example: Deflection δ = 5*w*L⁴/(384*E*I)
    
    Returns:
    --------
    dict
        Contains deflection function, Jacobian, and example
    """
    def deflection(params):
        """
        Compute beam deflection.
        
        Parameters:
        -----------
        params : array [b, h, L]
            b: width (m)
            h: height (m)
            L: span (m)
        
        Returns:
        --------
        float
            Deflection (mm)
        """
        b, h, L = params[0], params[1], params[2]
        w = 10  # Distributed load (kN/m)
        E = 200e3  # Young's modulus (MPa)
        I = b * h**3 / 12
        
        if I == 0:
            return 1e10
        
        delta = 5 * w * L**4 / (384 * E * I)
        return delta * 1000  # Convert to mm
    
    def deflection_jacobian(params):
        """
        Jacobian of deflection.
        
        Returns:
        --------
        array
            [∂δ/∂b, ∂δ/∂h, ∂δ/∂L]
        """
        from .calculus import gradient
        return gradient(deflection, params)
    
    # Example
    params = np.array([0.3, 0.5, 10.0])  # b=0.3m, h=0.5m, L=10m
    delta_val = deflection(params)
    jac_val = deflection_jacobian(params)
    
    return {
        'deflection_fn': deflection,
        'jacobian_fn': deflection_jacobian,
        'example': {
            'params': params,
            'deflection': delta_val,
            'jacobian': jac_val,
            'sensitivity': {
                'width': f'{jac_val[0]:.2f} mm/m',
                'height': f'{jac_val[1]:.2f} mm/m',
                'span': f'{jac_val[2]:.2f} mm/m'
            }
        }
    }


def cost_function_gradient():
    """
    Compute cost function gradient for optimization.
    
    Example: Total cost = material_cost + labor_cost
    
    Returns:
    --------
    dict
        Contains cost function, gradient, and optimization example
    """
    def total_cost(params):
        """
        Compute total project cost.
        
        Parameters:
        -----------
        params : array [fc, fy]
            fc: concrete strength (MPa)
            fy: steel yield strength (MPa)
        
        Returns:
        --------
        float
            Total cost ($)
        """
        fc, fy = params[0], params[1]
        
        # Material costs (higher strength = higher cost)
        concrete_cost_per_mpa = 2.0  # $/MPa per m³
        steel_cost_per_mpa = 5.0     # $/MPa per ton
        
        # Volume/weight requirements (inversely proportional to strength)
        concrete_volume = 1000 / fc  # m³
        steel_weight = 50 / fy       # tons
        
        # Total cost
        cost = (50 + concrete_cost_per_mpa * fc) * concrete_volume + \
               (500 + steel_cost_per_mpa * fy) * steel_weight
        
        return cost
    
    def cost_gradient(params):
        """
        Gradient of cost function.
        
        Returns:
        --------
        array
            [∂C/∂fc, ∂C/∂fy]
        """
        from .calculus import gradient
        return gradient(total_cost, params)
    
    # Optimization example using gradient descent
    from .calculus import gradient_descent_step
    
    params_init = np.array([30.0, 350.0])  # Initial guess
    params_opt = params_init.copy()
    
    history = [params_opt.copy()]
    cost_history = [total_cost(params_opt)]
    
    for _ in range(50):
        params_opt = gradient_descent_step(total_cost, params_opt, learning_rate=0.5)
        history.append(params_opt.copy())
        cost_history.append(total_cost(params_opt))
    
    return {
        'cost_fn': total_cost,
        'gradient_fn': cost_gradient,
        'optimization': {
            'initial_params': params_init,
            'initial_cost': cost_history[0],
            'optimal_params': params_opt,
            'optimal_cost': cost_history[-1],
            'history': np.array(history),
            'cost_history': cost_history,
            'savings': cost_history[0] - cost_history[-1]
        }
    }


def sensitivity_analysis_example():
    """
    Comprehensive sensitivity analysis for a CE problem.
    
    Returns:
    --------
    dict
        Sensitivity analysis results
    """
    # Example: Bridge pier design
    # Variables: diameter (D), height (H)
    # Outputs: cost, stability factor
    
    def cost(params):
        D, H = params[0], params[1]
        volume = np.pi * (D/2)**2 * H
        return 500 * volume  # $500 per m³
    
    def stability(params):
        D, H = params[0], params[1]
        # Simplified stability factor
        return D / H  # Should be > 0.1 for stability
    
    def multi_output(params):
        """Combined output [cost, stability]"""
        return np.array([cost(params), stability(params)])
    
    # Compute Jacobian
    from .calculus import jacobian
    
    params = np.array([2.0, 15.0])  # D=2m, H=15m
    J = jacobian(multi_output, params)
    
    return {
        'cost': cost(params),
        'stability': stability(params),
        'jacobian': J,
        'interpretation': {
            'dCost_dD': f'{J[0, 0]:.2f} $/m',
            'dCost_dH': f'{J[0, 1]:.2f} $/m',
            'dStability_dD': f'{J[1, 0]:.4f} per m',
            'dStability_dH': f'{J[1, 1]:.4f} per m'
        }
    }
