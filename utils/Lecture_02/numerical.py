"""
Numerical computing demonstrations for Linear Algebra concepts.

This module provides demonstrations of floating-point arithmetic,
memory usage, conditioning, and numerical edge cases.
"""

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


def demonstrate_floating_point():
    """
    Demonstrate IEEE 754 floating-point representation and precision issues.
    
    Returns:
    --------
    results : dict
        Dictionary containing various floating-point demonstrations
    """
    results = {}
    
    # Machine epsilon
    results['machine_epsilon_32'] = np.finfo(np.float32).eps
    results['machine_epsilon_64'] = np.finfo(np.float64).eps
    
    # Precision demonstration
    results['float32_precision'] = {
        '1.0 + 1e-7': np.float32(1.0) + np.float32(1e-7) == np.float32(1.0),
        '1.0 + 1e-8': np.float32(1.0) + np.float32(1e-8) == np.float32(1.0),
    }
    
    results['float64_precision'] = {
        '1.0 + 1e-15': np.float64(1.0) + np.float64(1e-15) == np.float64(1.0),
        '1.0 + 1e-16': np.float64(1.0) + np.float64(1e-16) == np.float64(1.0),
        '1.0 + 1e-17': np.float64(1.0) + np.float64(1e-17) == np.float64(1.0),
    }
    
    # Catastrophic cancellation
    a = np.float32(1.0)
    b = np.float32(1e-8)
    results['catastrophic_cancellation'] = {
        'a': a,
        'b': b,
        '(a + b) - a': (a + b) - a,
        'expected': b,
        'relative_error': abs(((a + b) - a) - b) / b if b != 0 else np.inf
    }
    
    # Associativity failure
    x, y, z = 1e20, 1.0, -1e20
    results['associativity'] = {
        '(x + y) + z': (x + y) + z,
        'x + (y + z)': x + (y + z),
        'equal': (x + y) + z == x + (y + z)
    }
    
    # Special values
    results['special_values'] = {
        'inf': np.inf,
        '-inf': -np.inf,
        'nan': np.nan,
        'inf + (-inf)': np.inf + (-np.inf),
        '0 / 0': np.float64(0.0) / np.float64(0.0),  # Returns nan (IEEE 754)
        '1 / 0': 'Would raise ZeroDivisionError in Python (use np.inf instead)',
    }
    
    # Print results
    print("=" * 60)
    print("FLOATING-POINT ARITHMETIC DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. Machine Epsilon (smallest ε where 1 + ε ≠ 1):")
    print(f"   float32: {results['machine_epsilon_32']:.2e}")
    print(f"   float64: {results['machine_epsilon_64']:.2e}")
    
    print("\n2. Precision Limits (does 1 + x equal 1?):")
    print("   float32:")
    for expr, result in results['float32_precision'].items():
        print(f"      {expr}: {result}")
    print("   float64:")
    for expr, result in results['float64_precision'].items():
        print(f"      {expr}: {result}")
    
    print("\n3. Catastrophic Cancellation:")
    cc = results['catastrophic_cancellation']
    print(f"   a = {cc['a']}, b = {cc['b']}")
    print(f"   (a + b) - a = {cc['(a + b) - a']}")
    print(f"   Expected: {cc['expected']}")
    print(f"   Relative error: {cc['relative_error']:.2%}")
    
    print("\n4. Associativity Failure:")
    assoc = results['associativity']
    print(f"   (1e20 + 1.0) + (-1e20) = {assoc['(x + y) + z']}")
    print(f"   1e20 + (1.0 + (-1e20)) = {assoc['x + (y + z)']}")
    print(f"   Are they equal? {assoc['equal']}")
    
    print("\n" + "=" * 60)
    
    return results


def memory_usage_table():
    """
    Create a table showing memory usage for different data types.
    
    Returns:
    --------
    df : pandas DataFrame
        Memory usage comparison table
    """
    dtypes = [
        ('bool', np.bool_),
        ('int8', np.int8),
        ('int16', np.int16),
        ('int32', np.int32),
        ('int64', np.int64),
        ('float16', np.float16),
        ('float32', np.float32),
        ('float64', np.float64),
        ('complex64', np.complex64),
        ('complex128', np.complex128),
    ]
    
    data = []
    for name, dtype in dtypes:
        # Create a sample array
        arr = np.array([1], dtype=dtype)
        
        # Get info
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            min_val = info.min
            max_val = info.max
            precision = 'N/A'
        elif np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            min_val = info.min
            max_val = info.max
            precision = info.precision
        elif np.issubdtype(dtype, np.complexfloating):
            info = np.finfo(dtype)
            min_val = 'Complex'
            max_val = 'Complex'
            precision = info.precision
        else:  # bool
            min_val = False
            max_val = True
            precision = 'N/A'
        
        data.append({
            'Data Type': name,
            'Bytes': arr.itemsize,
            'Bits': arr.itemsize * 8,
            'Min Value': str(min_val)[:20],
            'Max Value': str(max_val)[:20],
            'Precision': precision
        })
    
    df = pd.DataFrame(data)
    
    print("\n" + "=" * 80)
    print("MEMORY USAGE FOR DIFFERENT DATA TYPES")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Calculate memory for large arrays
    print("\nMemory for 1 million elements:")
    n = 1_000_000
    for name, dtype in dtypes[:8]:  # Skip complex for brevity
        arr = np.zeros(n, dtype=dtype)
        mem_mb = arr.nbytes / (1024 ** 2)
        print(f"   {name:10s}: {mem_mb:8.2f} MB")
    
    return df


def condition_number_demo(matrix_type='random', size=5, scale_factor=1000):
    """
    Demonstrate the effect of condition number on solution accuracy.
    
    Parameters:
    -----------
    matrix_type : str
        Type of matrix: 'random', 'hilbert', 'scaled'
    size : int
        Matrix size
    scale_factor : float
        Scaling factor for ill-conditioning
    
    Returns:
    --------
    results : dict
        Results of the demonstration
    """
    # Create matrix based on type
    if matrix_type == 'hilbert':
        # Hilbert matrix (notoriously ill-conditioned)
        A = np.array([[1.0 / (i + j + 1) for j in range(size)] for i in range(size)])
    elif matrix_type == 'scaled':
        # Diagonal matrix with varying scales
        A = np.diag([1.0] + [1.0 / scale_factor] * (size - 1))
    else:  # random
        A = np.random.randn(size, size)
        A = A @ A.T  # Make symmetric positive definite
    
    # True solution
    x_true = np.ones(size)
    b = A @ x_true
    
    # Solve
    x_computed = np.linalg.solve(A, b)
    
    # Add small perturbation to b
    perturbation = 1e-10 * np.random.randn(size)
    b_perturbed = b + perturbation
    x_perturbed = np.linalg.solve(A, b_perturbed)
    
    # Calculate condition number
    cond_num = np.linalg.cond(A)
    
    # Calculate errors
    b_error = np.linalg.norm(perturbation) / np.linalg.norm(b)
    x_error = np.linalg.norm(x_perturbed - x_true) / np.linalg.norm(x_true)
    
    # Theoretical bound
    theoretical_bound = cond_num * b_error
    
    results = {
        'matrix': A,
        'condition_number': cond_num,
        'x_true': x_true,
        'x_computed': x_computed,
        'x_perturbed': x_perturbed,
        'relative_input_error': b_error,
        'relative_output_error': x_error,
        'theoretical_bound': theoretical_bound,
        'amplification_factor': x_error / b_error if b_error > 0 else 0
    }
    
    # Print results
    print("\n" + "=" * 70)
    print(f"CONDITION NUMBER DEMONSTRATION ({matrix_type.upper()} MATRIX)")
    print("=" * 70)
    print(f"\nMatrix size: {size} × {size}")
    print(f"Condition number: {cond_num:.2e}")
    print(f"\nRelative input error (||δb|| / ||b||): {b_error:.2e}")
    print(f"Relative output error (||δx|| / ||x||): {x_error:.2e}")
    print(f"Amplification factor: {results['amplification_factor']:.2f}")
    print(f"Theoretical bound (κ × input error): {theoretical_bound:.2e}")
    print(f"\nInterpretation:")
    if cond_num < 10:
        print("   ✓ Well-conditioned: Small input errors → small output errors")
    elif cond_num < 1000:
        print("   ⚠ Moderately conditioned: Input errors may be amplified")
    else:
        print("   ✗ Ill-conditioned: Small input errors → large output errors!")
    print("=" * 70)
    
    return results


def near_singular_demo():
    """
    Demonstrate behavior near singular matrices (rank deficiency).
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Full rank matrix
    A1 = np.array([[2, 1], [1, 2]])
    rank1 = np.linalg.matrix_rank(A1)
    cond1 = np.linalg.cond(A1)
    
    axes[0, 0].imshow(A1, cmap='RdBu_r', aspect='auto')
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, f'{A1[i, j]:.1f}', ha="center", va="center",
                          color="black", fontsize=14, fontweight='bold')
    axes[0, 0].set_title(f'Full Rank\nRank={rank1}, κ={cond1:.1f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    # 2. Nearly singular (small determinant)
    A2 = np.array([[1, 1], [1, 1.0001]])
    rank2 = np.linalg.matrix_rank(A2)
    cond2 = np.linalg.cond(A2)
    
    axes[0, 1].imshow(A2, cmap='RdBu_r', aspect='auto')
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{A2[i, j]:.4f}', ha="center", va="center",
                          color="black", fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Nearly Singular\nRank={rank2}, κ={cond2:.1e}',
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    # 3. Exactly singular
    A3 = np.array([[1, 2], [2, 4]])
    rank3 = np.linalg.matrix_rank(A3)
    det3 = np.linalg.det(A3)
    
    axes[1, 0].imshow(A3, cmap='RdBu_r', aspect='auto')
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{A3[i, j]:.1f}', ha="center", va="center",
                          color="black", fontsize=14, fontweight='bold')
    axes[1, 0].set_title(f'Singular\nRank={rank3}, det={det3:.1e}',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    # 4. Rank comparison visualization
    matrices = [A1, A2, A3]
    labels = ['Full Rank', 'Nearly Singular', 'Singular']
    cond_numbers = [cond1, cond2, 1e16]  # Use large value for singular
    
    axes[1, 1].bar(labels, np.log10(cond_numbers), color=['green', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('log₁₀(Condition Number)', fontsize=11)
    axes[1, 1].set_title('Condition Number Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add actual values on bars
    for i, (label, cond) in enumerate(zip(labels, cond_numbers)):
        axes[1, 1].text(i, np.log10(cond) + 0.5, f'{cond:.1e}',
                       ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Matrix Rank and Conditioning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("NEAR-SINGULAR MATRIX DEMONSTRATION")
    print("=" * 70)
    print("\n1. Full Rank Matrix:")
    print(f"   Rank: {rank1}, Condition Number: {cond1:.2f}")
    print("   → Well-conditioned, unique solution exists\n")
    
    print("2. Nearly Singular Matrix:")
    print(f"   Rank: {rank2}, Condition Number: {cond2:.2e}")
    print("   → Ill-conditioned, solution exists but unstable\n")
    
    print("3. Exactly Singular Matrix:")
    print(f"   Rank: {rank3}, Determinant: {det3:.2e}")
    print("   → Rank deficient, no unique solution (or no solution)")
    print("=" * 70)
    
    return fig, axes
