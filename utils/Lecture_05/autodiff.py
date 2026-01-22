"""
Automatic differentiation using forward and backward modes.

This module provides educational implementations of forward-mode and
backward-mode automatic differentiation.
"""

import numpy as np


class Variable:
    """
    Variable for forward-mode automatic differentiation using dual numbers.
    
    A dual number has the form: a + b*ε where ε² = 0
    Represents (value, derivative) pairs.
    
    Attributes:
    -----------
    value : float
        Function value
    derivative : float
        Derivative value
    """
    
    def __init__(self, value, derivative=0.0):
        """
        Initialize a dual number variable.
        
        Parameters:
        -----------
        value : float
            Value part
        derivative : float
            Derivative part (default 0 for constants)
        """
        self.value = float(value)
        self.derivative = float(derivative)
    
    def __repr__(self):
        return f"Variable(value={self.value:.4f}, deriv={self.derivative:.4f})"
    
    def __add__(self, other):
        """Addition: (a + b*ε) + (c + d*ε) = (a+c) + (b+d)*ε"""
        if isinstance(other, Variable):
            return Variable(
                self.value + other.value,
                self.derivative + other.derivative
            )
        else:
            return Variable(self.value + other, self.derivative)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtraction"""
        if isinstance(other, Variable):
            return Variable(
                self.value - other.value,
                self.derivative - other.derivative
            )
        else:
            return Variable(self.value - other, self.derivative)
    
    def __rsub__(self, other):
        return Variable(other, 0.0) - self
    
    def __mul__(self, other):
        """Multiplication: (a + b*ε) * (c + d*ε) = ac + (ad + bc)*ε"""
        if isinstance(other, Variable):
            return Variable(
                self.value * other.value,
                self.value * other.derivative + self.derivative * other.value
            )
        else:
            return Variable(self.value * other, self.derivative * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Division: (a + b*ε) / (c + d*ε) = a/c + (bc - ad)/c²*ε"""
        if isinstance(other, Variable):
            return Variable(
                self.value / other.value,
                (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2)
            )
        else:
            return Variable(self.value / other, self.derivative / other)
    
    def __rtruediv__(self, other):
        return Variable(other, 0.0) / self
    
    def __pow__(self, n):
        """Power: (a + b*ε)^n = a^n + n*a^(n-1)*b*ε"""
        return Variable(
            self.value ** n,
            n * (self.value ** (n - 1)) * self.derivative
        )
    
    def __neg__(self):
        """Negation"""
        return Variable(-self.value, -self.derivative)
    
    def exp(self):
        """Exponential: exp(a + b*ε) = exp(a) + exp(a)*b*ε"""
        exp_val = np.exp(self.value)
        return Variable(exp_val, exp_val * self.derivative)
    
    def log(self):
        """Logarithm: log(a + b*ε) = log(a) + (b/a)*ε"""
        return Variable(np.log(self.value), self.derivative / self.value)
    
    def sin(self):
        """Sine: sin(a + b*ε) = sin(a) + cos(a)*b*ε"""
        return Variable(np.sin(self.value), np.cos(self.value) * self.derivative)
    
    def cos(self):
        """Cosine: cos(a + b*ε) = cos(a) - sin(a)*b*ε"""
        return Variable(np.cos(self.value), -np.sin(self.value) * self.derivative)


def forward_mode_ad(f, x, i=0):
    """
    Compute derivative using forward-mode automatic differentiation.
    
    Parameters:
    -----------
    f : callable
        Function that accepts Variable objects
    x : array-like
        Point at which to compute derivative
    i : int
        Index of variable to differentiate with respect to
    
    Returns:
    --------
    tuple
        (function_value, derivative)
    
    Example:
    --------
    >>> def f(x, y):
    ...     return x**2 + 3*y
    >>> forward_mode_ad(lambda vars: f(vars[0], vars[1]), [2.0, 3.0], i=0)
    (13.0, 4.0)  # f(2,3) = 13, ∂f/∂x = 4
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Create Variable objects
    # Set derivative to 1 for variable i, 0 for others
    vars = []
    for j in range(n):
        if j == i:
            vars.append(Variable(x[j], derivative=1.0))
        else:
            vars.append(Variable(x[j], derivative=0.0))
    
    # Evaluate function
    result = f(vars)
    
    return result.value, result.derivative


def backward_mode_ad(f, x):
    """
    Compute gradient using backward-mode automatic differentiation.
    
    This is a simplified implementation using the computation graph.
    
    Parameters:
    -----------
    f : callable
        Function that builds a computation graph
    x : array-like
        Point at which to compute gradient
    
    Returns:
    --------
    tuple
        (function_value, gradient_vector)
    
    Note:
    -----
    Function f should accept a list of Node objects and return a Node.
    """
    from .computation_graph import ComputationGraph, Node
    
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Create computation graph
    graph = ComputationGraph()
    
    # Create input nodes
    input_nodes = []
    for i, val in enumerate(x):
        node = graph.add_input(val, name=f"x{i}")
        input_nodes.append(node)
    
    # Build computation
    output = f(input_nodes)
    graph.set_output(output)
    
    # Forward pass
    value = graph.forward()
    
    # Backward pass
    graph.backward()
    
    # Extract gradients
    gradient = np.array([node.grad for node in input_nodes])
    
    return value, gradient


def tape_based_autodiff(operations):
    """
    Tape-based automatic differentiation.
    
    Records operations on a "tape" and plays them backward.
    
    Parameters:
    -----------
    operations : list
        List of (operation, inputs, output) tuples
    
    Returns:
    --------
    dict
        Gradients for all variables
    
    Example:
    --------
    This is a conceptual implementation showing how modern
    frameworks like PyTorch work internally.
    """
    # This is a simplified conceptual implementation
    # Real implementations use more sophisticated tape structures
    
    gradients = {}
    
    # Initialize output gradient to 1
    output_var = operations[-1][2]
    gradients[output_var] = 1.0
    
    # Traverse tape backward
    for op, inputs, output in reversed(operations):
        if output not in gradients:
            continue
        
        output_grad = gradients[output]
        
        # Distribute gradient to inputs based on operation
        if op == 'add':
            for inp in inputs:
                gradients[inp] = gradients.get(inp, 0.0) + output_grad
        
        elif op == 'mul':
            # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            a, b = inputs
            gradients[a] = gradients.get(a, 0.0) + output_grad * b
            gradients[b] = gradients.get(b, 0.0) + output_grad * a
        
        elif op == 'pow':
            base, exp = inputs
            # ∂(a^n)/∂a = n*a^(n-1)
            gradients[base] = gradients.get(base, 0.0) + output_grad * exp * (base ** (exp - 1))
    
    return gradients


def compare_autodiff_methods(f, x):
    """
    Compare forward-mode, backward-mode, and numerical differentiation.
    
    Parameters:
    -----------
    f : callable
        Function (for forward-mode: accepts Variables, for backward: accepts Nodes)
    x : array-like
        Point to evaluate
    
    Returns:
    --------
    dict
        Comparison results
    """
    from .calculus import gradient as numerical_gradient
    
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Numerical gradient
    def f_numerical(x_arr):
        # Convert to Variables for evaluation
        vars = [Variable(val) for val in x_arr]
        result = f(vars)
        return result.value if isinstance(result, Variable) else result
    
    num_grad = numerical_gradient(f_numerical, x)
    
    # Forward-mode AD (compute each partial separately)
    forward_grad = np.zeros(n)
    for i in range(n):
        _, deriv = forward_mode_ad(f, x, i)
        forward_grad[i] = deriv
    
    # Backward-mode AD
    def f_backward(nodes):
        # Reconstruct function using nodes
        return f(nodes)
    
    try:
        _, backward_grad = backward_mode_ad(f_backward, x)
    except:
        backward_grad = None
    
    return {
        'numerical': num_grad,
        'forward_mode': forward_grad,
        'backward_mode': backward_grad,
        'point': x
    }
