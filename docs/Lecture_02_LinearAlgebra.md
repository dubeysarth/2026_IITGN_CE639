# CE639 Lecture 02: Linear Algebra for ML
## Interactive Jupyter Notebook Documentation

**Course:** CE639 - AI for Civil Engineering  
**Instructors:** Dr. Udit Bhatia & Dr. Sushobhan Sen  
**Tutor:** Sarth Dubey  
**Institution:** IIT Gandhinagar

---

## 📚 Overview

This comprehensive Jupyter Notebook teaches Linear Algebra fundamentals in the context of Machine Learning and Civil Engineering applications. The notebook features interactive widgets, animations, and real-world CE examples to enhance student engagement and understanding.

### Learning Objectives

By the end of this notebook, students will be able to:
- Differentiate between different types of tensors and understand their computer storage
- Explain projections, rank, and conditioning in the context of ML
- Define and calculate eigenvalues and singular values
- Apply these concepts to Civil Engineering problems

---

## 📁 File Structure

```
CE639/
├── Lecture_02_LinearAlgebra.ipynb          # Main notebook (31 KB, 1000+ lines)
└── utils/Lecture_02/                     # Helper utilities module
    ├── __init__.py                       # Package initialization
    ├── visualizations.py                 # Plotting & animation functions
    ├── widgets.py                        # Interactive ipywidgets
    ├── ce_examples.py                    # Civil Engineering examples
    └── numerical.py                      # Numerical computing demos
```

### Module Breakdown

| Module | Size | Functions | Purpose |
|--------|------|-----------|---------|
| `visualizations.py` | 550+ lines | 7 functions | Vector/matrix plots, animations |
| `widgets.py` | 350+ lines | 5 functions | Interactive sliders and controls |
| `ce_examples.py` | 400+ lines | 5 functions | Structural analysis examples |
| `numerical.py` | 450+ lines | 4 functions | Floating-point demonstrations |

---

## 📖 Notebook Sections

### Section 1: Setup & Installation
**Purpose:** Environment configuration and dependency management

- Auto-detects Google Colab vs local environment
- Installs required packages: `numpy`, `matplotlib`, `ipywidgets`, `pandas`
- Imports all helper utilities from `utils.Lecture_02`
- Configures matplotlib for inline plotting

---

### Section 2: Tensors - The Building Blocks

**Mathematical Foundation:**

A **scalar** is a 0-dimensional tensor (single number).  
A **vector** $\mathbf{x} \in \mathbb{R}^n$ is a 1-dimensional tensor:

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$$

A **matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a 2-dimensional tensor:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Content Covered:**
- Scalar data types (int, float, bool, complex)
- Vector operations: addition, scalar multiplication, dot product
- Matrix operations: transpose, multiplication
- Higher-order tensors (3D, 4D)

**Visualizations:**
- 2D and 3D vector arrow plots
- Matrix heatmaps with color gradients
- Stress tensor visualization

**CE Examples:**
- Displacement vectors in structural analysis
- Structural stiffness matrix $\mathbf{K}$ (symmetric, positive definite)
- Stress tensor (3×3 symmetric matrix)

**Interactive Widget:**
- Vector scaling slider (visualize scalar multiplication)

---

### Section 3: Norms - Measuring Magnitude

**Mathematical Foundation:**

The **$L_p$ norm** of a vector $\mathbf{x}$ is defined as:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

Special cases:
- **$L_1$ (Manhattan):** $\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$
- **$L_2$ (Euclidean):** $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$
- **$L_\infty$ (Max):** $\|\mathbf{x}\|_\infty = \max_i |x_i|$

**Content Covered:**
- L1, L2, L∞ norms and their interpretations
- p-norm generalization
- Unit ball visualization for different p-values
- Applications in ML (LASSO, Ridge regression)

**Visualizations:**
- Unit balls for p = 0.5, 1, 2, ∞
- Error comparison plots

**CE Examples:**
- Structural deflection error measurement
- Predicted vs actual deflection comparison

**Interactive Widget:**
- Norm explorer (adjust p-value from 0.5 to 10, see unit ball morph)

**Edge Cases:**
- Zero vectors
- Sparse vs dense vectors
- L0 "norm" (sparsity measure)

---

### Section 4: Computer Memory & Floating Point

**Mathematical Foundation:**

IEEE 754 floating-point representation:

$$\text{value} = (-1)^S \times (1.\text{Mantissa}) \times 2^{\text{Exponent}}$$

For **float32**: 1 sign bit + 8 exponent bits + 23 mantissa bits  
For **float64**: 1 sign bit + 11 exponent bits + 52 mantissa bits

**Machine epsilon** $\epsilon_{\text{mach}}$: smallest number where $1 + \epsilon \neq 1$

**Content Covered:**
- IEEE 754 representation (float32 vs float64)
- Machine epsilon demonstration
- Catastrophic cancellation
- Associativity failure in floating-point arithmetic
- Memory usage tables for different data types

**Demonstrations:**
- `demonstrate_floating_point()` - precision limits
- `memory_usage_table()` - dtype comparison
- Large matrix memory estimation

**Key Insights:**
- Why 64-bit precision is better than 32-bit
- Numerical precision awareness in ML training

---

### Section 5: Projections

**Mathematical Foundation:**

The **projection** of vector $\mathbf{x}$ onto vector $\mathbf{w}$ is:

$$\text{proj}_\mathbf{w}(\mathbf{x}) = \frac{\mathbf{w}^T \mathbf{x}}{\mathbf{w}^T \mathbf{w}} \mathbf{w}$$

The **projection matrix** onto the column space of $\mathbf{A}$ is:

$$\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

This is the **pseudo-inverse** used in linear regression.

**Content Covered:**
- Vector-to-vector projection formula
- Matrix projection (pseudo-inverse)
- Geometric interpretation
- Connection to dimensionality reduction

**Visualizations:**
- Static projection plots with perpendicular components
- Animated projection (growing projection vector)

**CE Examples:**
- Force decomposition on beam (axial + transverse components)
- Right-angle indicator visualization

**Interactive Widget:**
- Projection explorer (4 sliders for $x_1, x_2, w_1, w_2$)

---

### Section 6: Matrix Rank

**Mathematical Foundation:**

The **rank** of a matrix $\mathbf{A}$ is the number of linearly independent rows or columns:

$$\text{rank}(\mathbf{A}) \leq \min(m, n) \quad \text{for } \mathbf{A} \in \mathbb{R}^{m \times n}$$

- **Full rank:** $\text{rank}(\mathbf{A}) = \min(m, n)$
- **Rank deficient:** $\text{rank}(\mathbf{A}) < \min(m, n)$

**Content Covered:**
- Rank definition (linear independence)
- Full rank vs rank deficient matrices
- Rank and data quality in ML
- Correlation and redundancy

**Demonstrations:**
- Full rank vs rank-deficient matrices
- Near-singular matrix visualization

**CE Examples:**
- Structural indeterminacy detection
- Determinate vs indeterminate structures
- Rigid body modes (rank deficiency)

**Data Quality:**
- Redundant features in datasets
- Correlation matrix analysis

---

### Section 7: Conditioning

**Mathematical Foundation:**

The **condition number** of a matrix $\mathbf{A}$ is:

$$\kappa(\mathbf{A}) = \|\mathbf{A}\|_2 \cdot \|\mathbf{A}^{-1}\|_2$$

Error amplification bound:

$$\frac{\|\delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot \frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|}$$

- **$\kappa \approx 1$:** Well-conditioned
- **$\kappa \gg 1$:** Ill-conditioned

**Content Covered:**
- Condition number definition and computation
- Well-conditioned vs ill-conditioned systems
- Error amplification in numerical solutions
- Impact on ML training convergence

**Demonstrations:**
- `condition_number_demo()` with Hilbert matrices
- Input error → output error amplification

**CE Examples:**
- Feature scaling impact (before/after)
- Unscaled features [0,1] vs [0,1000]
- Condition number improvement through normalization

**Interactive Widget:**
- Conditioning explorer (adjust scale ratio, noise level)

---

### Section 8: Eigenvalue Decomposition

**Mathematical Foundation:**

For a square matrix $\mathbf{A}$, if:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

then $\mathbf{v}$ is an **eigenvector** and $\lambda$ is an **eigenvalue**.

**Eigendecomposition:**

$$\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$$

where $\mathbf{\Lambda}$ is diagonal (eigenvalues) and columns of $\mathbf{V}$ are eigenvectors.

**Content Covered:**
- Eigenvalues and eigenvectors
- $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ interpretation
- Eigendecomposition formula
- Connection to PCA in ML

**Visualizations:**
- Animated linear transformation
- Eigenvectors remain in same direction (only scaled)

**CE Examples:**
- Structural vibration analysis (3-mass system)
- Natural frequencies: $\omega = \sqrt{\lambda}$
- Mode shape visualization
- Animated vibration modes (multiple modes)

**Verification:**
- $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ for each eigenpair

---

### Section 9: Singular Value Decomposition (SVD)

**Mathematical Foundation:**

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: Left singular vectors (orthonormal)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: Singular values (diagonal)
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: Right singular vectors (orthonormal)

**Relationship to eigenvalues:**

$$\sigma_i^2 = \lambda_i(\mathbf{A}^T\mathbf{A})$$

**Content Covered:**
- SVD formula and components
- Relationship to eigenvalues
- Truncated SVD for compression
- Applications in recommender systems

**Visualizations:**
- Image compression at different ranks
- Singular value spectrum (log scale)
- Cumulative energy plots

**CE Examples:**
- Sensor data compression (structural health monitoring)
- 10 sensors × 100 timesteps
- Energy capture analysis (95% threshold)

**Interactive Widget:**
- SVD rank slider (truncation rank 1 to max)
- Real-time compression ratio and error display

---

### Section 10: Summary & Resources

**Key Concepts Review:**
- Tensors: Scalars → Vectors → Matrices → Higher-order
- Norms: L1, L2, L∞ and their applications
- Computer memory and numerical precision
- Projections and dimensionality reduction
- Rank and linear independence
- Conditioning and numerical stability
- Eigendecomposition and principal directions
- SVD and matrix factorization

**Connections to Machine Learning:**

| Linear Algebra Concept | ML Application |
|------------------------|----------------|
| Vectors | Feature vectors, embeddings |
| Matrices | Datasets, weight matrices |
| Norms | Loss functions, regularization (L1/L2) |
| Projections | PCA, dimensionality reduction |
| Rank | Feature redundancy detection |
| Conditioning | Training stability, convergence |
| Eigenvalues | PCA, spectral clustering |
| SVD | Recommender systems, LSA, matrix completion |

**Further Reading:**
1. *Linear Algebra and Its Applications* by Gilbert Strang
2. *Matrix Computations* by Golub & Van Loan
3. *Deep Learning* by Goodfellow, Bengio & Courville (Chapter 2)
4. MIT OCW: Linear Algebra (18.06)
5. 3Blue1Brown: Essence of Linear Algebra (YouTube)

---

## 🎨 Interactive Features

### Widgets (5 total)

1. **Vector Scaling Widget**
   - Scale factor slider (-3 to 3)
   - Real-time vector visualization

2. **Norm Explorer Widget**
   - p-value slider (0.5 to 10)
   - Unit ball morphing visualization

3. **Projection Widget**
   - 4 sliders: $x_1, x_2, w_1, w_2$
   - Live projection and perpendicular component display
   - Angle and magnitude calculations

4. **SVD Rank Widget**
   - Rank slider (1 to max rank)
   - Compression ratio display
   - Relative error calculation
   - Singular value spectrum

5. **Conditioning Widget**
   - Scale ratio slider (1 to 100)
   - Noise level slider (0 to 0.5)
   - Error amplification visualization

### Animations (3 total)

1. **Vector Projection Animation**
   - Growing projection vector
   - Perpendicular component reveal
   - 60 frames, smooth transitions

2. **Eigenvalue Transformation Animation**
   - Grid of vectors transforming
   - Eigenvectors highlighted in red
   - 120 frames showing rotation + scaling

3. **Vibration Mode Animation**
   - Structural mode shapes oscillating
   - Physical displacement visualization
   - Mode shape components

---

## 🏗️ Civil Engineering Examples

### 1. Displacement Vectors
- 2D and 3D displacement visualization
- Node displacement in structural analysis

### 2. Structural Stiffness Matrix
- Symmetric positive definite matrix
- Tridiagonal structure for spring systems
- Condition number analysis

### 3. Stress Tensor
- 3×3 symmetric stress tensor
- Visualization as heatmap
- Equilibrium verification

### 4. Deflection Error Measurement
- Predicted vs actual deflections
- L1, L2, L∞ error metrics
- Visual error comparison

### 5. Force Decomposition on Beam
- Axial and transverse components
- Projection onto beam axes
- Right-angle indicator

### 6. Structural Indeterminacy Detection
- Determinate vs indeterminate structures
- Rank deficiency and rigid body modes
- Stiffness matrix singularity

### 7. Vibration Mode Analysis
- 3-mass spring system
- Natural frequency calculation
- Mode shape visualization and animation

### 8. Sensor Data Compression
- Structural health monitoring data
- SVD-based compression
- Energy capture analysis

---

## 🧪 Testing & Usage

### For Google Colab

1. **Upload Files:**
   ```
   Upload to Colab:
   - Lecture_02_LinearAlgebra.ipynb
   - utils/Lecture_02/ (entire directory)
   ```

2. **Run All Cells:**
   - Runtime → Run all

3. **Expected Behavior:**
   - Setup cell installs packages automatically
   - All imports succeed without errors
   - Visualizations render inline
   - Widgets display interactive sliders
   - Animations play as HTML5 video

### Verification Checklist

- ✓ All imports successful (no `ModuleNotFoundError`)
- ✓ Vector plots render correctly
- ✓ Matrix heatmaps display with colorbars
- ✓ Interactive widgets respond to slider changes
- ✓ Animations play smoothly (may take time to render)
- ✓ CE examples execute without errors
- ✓ No division by zero or numerical errors
- ✓ Memory usage tables display properly
- ✓ SVD compression shows visual differences

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| **Main Notebook** | 31 KB, 1000+ lines |
| **Helper Modules** | 54 KB total (5 files) |
| **Total Sections** | 10 |
| **Code Cells** | ~80 |
| **Markdown Cells** | ~40 |
| **Visualizations** | 20+ |
| **Interactive Widgets** | 5 |
| **Animations** | 3 |
| **CE Examples** | 8 |
| **Helper Functions** | 25+ |

---

## 🎓 Pedagogical Approach

1. **Progressive Complexity**
   - Start simple: Scalars → Vectors → Matrices → Tensors
   - Build understanding incrementally

2. **Visual First**
   - Every concept has visualization
   - Geometric intuition before algebra

3. **Interactive Exploration**
   - Widgets for hands-on learning
   - Immediate feedback on parameter changes

4. **CE Context**
   - Real-world civil engineering applications
   - Motivation through practical examples

5. **Edge Cases**
   - Stress testing with boundary conditions
   - Numerical precision awareness

6. **Verification**
   - Mathematical proofs
   - Numerical checks for correctness

---

## 🔧 Helper Functions Reference

### `visualizations.py`

```python
plot_vector_2d(vectors, labels, colors, title, xlim, ylim)
plot_vector_3d(vectors, labels, colors, title, xlim, ylim, zlim)
plot_matrix_heatmap(matrix, title, cmap, figsize, annot, fmt)
plot_norm_unit_balls(p_values, figsize)
animate_projection(vector, onto_vector, n_frames, figsize)
animate_eigen_transform(matrix, n_frames, figsize)
plot_svd_compression(image_array, ranks, figsize)
```

### `widgets.py`

```python
vector_scaling_widget(initial_vector)
norm_explorer_widget()
projection_widget(vector_x, vector_w)
svd_rank_widget(image_array)
condition_number_widget()
```

### `ce_examples.py`

```python
create_stiffness_matrix(n_dof, k_values)
create_vibration_system(n_masses, mass_values, k_values)
simulate_structural_modes(eigenvectors, n_frames, mode_index)
generate_sensor_data(n_sensors, n_timesteps, n_modes, noise_level)
create_force_decomposition(force_magnitude, force_angle_deg, beam_angle_deg)
```

### `numerical.py`

```python
demonstrate_floating_point()
memory_usage_table()
condition_number_demo(matrix_type, size, scale_factor)
near_singular_demo()
```

---

## ✅ Alignment with Course Objectives

All lecture objectives covered:
- ✓ Differentiate tensor types and computer storage
- ✓ Explain projections, rank, and conditioning
- ✓ Define and calculate eigenvalues and singular values
- ✓ Apply concepts to Civil Engineering problems

---

## 📝 Notes

- **Dependencies:** All required packages auto-install on Colab
- **Cell Format:** Uses `#%%` dividers for easy conversion to `.ipynb`
- **Compatibility:** Tested on Google Colab and local Jupyter environments
- **Performance:** Animations may take 10-30 seconds to render
- **Interactivity:** Requires `ipywidgets` for interactive sliders

---

*Documentation prepared for CE639: AI for Civil Engineering*  
*IIT Gandhinagar*
