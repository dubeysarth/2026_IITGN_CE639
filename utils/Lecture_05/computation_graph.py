"""
Computation graph for automatic differentiation.

This module provides a simple computation graph implementation for
demonstrating forward and backward passes in autodiff.
"""

import numpy as np
from collections import defaultdict


class Node:
    """
    Node in a computation graph.
    
    Attributes:
    -----------
    value : float or array
        Forward pass value
    grad : float or array
        Backward pass gradient (adjoint)
    name : str
        Node identifier
    parents : list
        Parent nodes
    operation : str
        Operation that created this node
    """
    
    def __init__(self, value, name=None, parents=None, operation=None):
        """
        Initialize a computation graph node.
        
        Parameters:
        -----------
        value : float or array
            Node value
        name : str, optional
            Node name
        parents : list, optional
            Parent nodes
        operation : str, optional
            Operation name
        """
        self.value = value
        self.grad = 0.0
        self.name = name or f"node_{id(self)}"
        self.parents = parents or []
        self.operation = operation or "input"
        self.local_gradients = {}  # Store ∂self/∂parent
    
    def __repr__(self):
        return f"Node({self.name}, value={self.value:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        """Addition operation."""
        if not isinstance(other, Node):
            other = Node(other, name=f"const_{other}")
        
        result = Node(
            value=self.value + other.value,
            name=f"({self.name}+{other.name})",
            parents=[self, other],
            operation="add"
        )
        
        # Local gradients: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        result.local_gradients[self] = 1.0
        result.local_gradients[other] = 1.0
        
        return result
    
    def __mul__(self, other):
        """Multiplication operation."""
        if not isinstance(other, Node):
            other = Node(other, name=f"const_{other}")
        
        result = Node(
            value=self.value * other.value,
            name=f"({self.name}*{other.name})",
            parents=[self, other],
            operation="mul"
        )
        
        # Local gradients: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        result.local_gradients[self] = other.value
        result.local_gradients[other] = self.value
        
        return result
    
    def __pow__(self, n):
        """Power operation."""
        result = Node(
            value=self.value ** n,
            name=f"({self.name}^{n})",
            parents=[self],
            operation=f"pow{n}"
        )
        
        # Local gradient: ∂(a^n)/∂a = n*a^(n-1)
        result.local_gradients[self] = n * (self.value ** (n - 1))
        
        return result
    
    def __sub__(self, other):
        """Subtraction operation."""
        if not isinstance(other, Node):
            other = Node(other, name=f"const_{other}")
        return self + (other * Node(-1, name="neg1"))
    
    def __truediv__(self, other):
        """Division operation."""
        if not isinstance(other, Node):
            other = Node(other, name=f"const_{other}")
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rsub__(self, other):
        return Node(other, name=f"const_{other}") - self
    
    def __rtruediv__(self, other):
        return Node(other, name=f"const_{other}") / self


class ComputationGraph:
    """
    Computation graph for automatic differentiation.
    
    Manages forward and backward passes through a DAG.
    """
    
    def __init__(self):
        """Initialize empty computation graph."""
        self.nodes = []
        self.inputs = []
        self.output = None
    
    def add_input(self, value, name=None):
        """
        Add input node to graph.
        
        Parameters:
        -----------
        value : float
            Input value
        name : str, optional
            Input name
        
        Returns:
        --------
        Node
            Input node
        """
        node = Node(value, name=name)
        self.inputs.append(node)
        self.nodes.append(node)
        return node
    
    def set_output(self, node):
        """
        Set output node.
        
        Parameters:
        -----------
        node : Node
            Output node
        """
        self.output = node
        self._collect_nodes(node)
    
    def _collect_nodes(self, node):
        """Collect all nodes in topological order."""
        visited = set()
        self.nodes = []
        
        def visit(n):
            if n in visited:
                return
            visited.add(n)
            for parent in n.parents:
                visit(parent)
            self.nodes.append(n)
        
        visit(node)
    
    def forward(self):
        """
        Perform forward pass.
        
        Returns:
        --------
        float
            Output value
        """
        # Values already computed during graph construction
        return self.output.value
    
    def backward(self):
        """
        Perform backward pass (backpropagation).
        
        Computes gradients for all nodes with respect to output.
        
        Returns:
        --------
        dict
            Gradients for all input nodes
        """
        # Reset gradients
        for node in self.nodes:
            node.grad = 0.0
        
        # Output gradient is 1 (∂output/∂output = 1)
        self.output.grad = 1.0
        
        # Traverse in reverse topological order
        for node in reversed(self.nodes):
            # Distribute gradient to parents
            for parent in node.parents:
                local_grad = node.local_gradients.get(parent, 0.0)
                parent.grad += node.grad * local_grad
        
        # Return gradients for inputs
        return self.get_gradients()
    
    def get_gradients(self):
        """
        Get gradients for all input nodes.
        
        Returns:
        --------
        dict
            Mapping from input names to gradients
        """
        return {node.name: node.grad for node in self.inputs}
    
    def visualize(self):
        """
        Get graph structure for visualization.
        
        Returns:
        --------
        dict
            Graph structure with nodes and edges
        """
        edges = []
        node_info = {}
        
        for node in self.nodes:
            node_info[node.name] = {
                'value': node.value,
                'grad': node.grad,
                'operation': node.operation
            }
            
            for parent in node.parents:
                edges.append((parent.name, node.name))
        
        return {
            'nodes': node_info,
            'edges': edges
        }


def forward_pass(graph):
    """
    Perform forward pass on computation graph.
    
    Parameters:
    -----------
    graph : ComputationGraph
        Computation graph
    
    Returns:
    --------
    float
        Output value
    """
    return graph.forward()


def backward_pass(graph):
    """
    Perform backward pass on computation graph.
    
    Parameters:
    -----------
    graph : ComputationGraph
        Computation graph
    
    Returns:
    --------
    dict
        Gradients for all inputs
    """
    graph.backward()
    return graph.get_gradients()


def create_simple_graph(x1_val, x2_val):
    """
    Create simple example graph: y = (x1 + x2)^2
    
    Parameters:
    -----------
    x1_val : float
        Value of x1
    x2_val : float
        Value of x2
    
    Returns:
    --------
    ComputationGraph
        Graph with inputs x1, x2 and output y
    """
    graph = ComputationGraph()
    
    # Create inputs
    x1 = graph.add_input(x1_val, name="x1")
    x2 = graph.add_input(x2_val, name="x2")
    
    # Build computation: y = (x1 + x2)^2
    z = x1 + x2
    z.name = "z"
    y = z ** 2
    y.name = "y"
    
    # Set output
    graph.set_output(y)
    
    return graph


def create_neural_network_graph(x_val, w1_val, b1_val, w2_val, b2_val):
    """
    Create simple neural network graph.
    
    Architecture: x -> w1*x + b1 -> sigmoid -> w2*a + b2 -> y
    
    Parameters:
    -----------
    x_val : float
        Input value
    w1_val : float
        First weight
    b1_val : float
        First bias
    w2_val : float
        Second weight
    b2_val : float
        Second bias
    
    Returns:
    --------
    ComputationGraph
        Neural network graph
    """
    graph = ComputationGraph()
    
    # Inputs
    x = graph.add_input(x_val, name="x")
    w1 = graph.add_input(w1_val, name="w1")
    b1 = graph.add_input(b1_val, name="b1")
    w2 = graph.add_input(w2_val, name="w2")
    b2 = graph.add_input(b2_val, name="b2")
    
    # First layer
    z1 = w1 * x + b1
    z1.name = "z1"
    
    # Sigmoid activation (approximated)
    # For simplicity, we'll use a linear approximation or just store the value
    # In practice, you'd implement sigmoid as a custom operation
    a1 = Node(1 / (1 + np.exp(-z1.value)), name="a1", parents=[z1], operation="sigmoid")
    a1.local_gradients[z1] = a1.value * (1 - a1.value)  # sigmoid derivative
    
    # Second layer
    z2 = w2 * a1 + b2
    z2.name = "z2"
    
    # Output
    graph.set_output(z2)
    
    return graph
