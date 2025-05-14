NumPy Mastery: Array Splitting, Linear Algebra, and Statistical Analysis
Course Overview
This course explores essential NumPy functionality for scientific computing and data analysis:

Array Splitting: Dividing arrays in different dimensions with np.split(), np.hsplit(), and np.vsplit()
Linear Algebra Fundamentals: Computing determinants, eigenvalues, and eigenvectors using np.linalg functions
Statistical Analysis: Calculating descriptive statistics, correlations, and covariances

Each module includes theoretical explanations, practical code examples, and exercises to reinforce learning.
Module 1: Array Splitting in NumPy
Introduction
Array splitting is a fundamental operation in NumPy that allows us to divide large arrays into smaller subarrays. This is particularly useful when dealing with batch processing, cross-validation in machine learning, or distributing computational workload.
1.1 Basic Array Splitting with np.split()
np.split() divides an array into multiple sub-arrays along a specified axis.
Syntax:
pythonnp.split(array, indices_or_sections, axis=0)
Where:

array: The NumPy array to be split
indices_or_sections: Can be either:

An integer indicating the number of equal-sized sub-arrays to create
A 1-D array of sorted integers indicating the indices where the array should be split


axis: The axis along which to split the array (default: 0)

Example 1: Splitting into equal parts
pythonimport numpy as np

# Create a 1D array
arr = np.arange(10)
print("Original array:", arr)

# Split into 5 equal parts
result = np.split(arr, 5)
print("Split into 5 equal parts:")
for i, subarr in enumerate(result):
    print(f"Subarray {i}:", subarr)
Output:
Original array: [0 1 2 3 4 5 6 7 8 9]
Split into 5 equal parts:
Subarray 0: [0 1]
Subarray 1: [2 3]
Subarray 2: [4 5]
Subarray 3: [6 7]
Subarray 4: [8 9]
Example 2: Splitting at specific indices
python# Split at indices 2, 5, and 8
result = np.split(arr, [2, 5, 8])
print("\nSplit at indices [2, 5, 8]:")
for i, subarr in enumerate(result):
    print(f"Subarray {i}:", subarr)
Output:
Split at indices [2, 5, 8]:
Subarray 0: [0 1]
Subarray 1: [2 3 4]
Subarray 2: [5 6 7]
Subarray 3: [8 9]
Example 3: Splitting a 2D array along different axes
python# Create a 2D array
arr_2d = np.arange(16).reshape(4, 4)
print("\nOriginal 2D array:")
print(arr_2d)

# Split along axis 0 (rows)
result_axis0 = np.split(arr_2d, 2, axis=0)
print("\nSplit along axis 0 (rows):")
for i, subarr in enumerate(result_axis0):
    print(f"Subarray {i}:")
    print(subarr)

# Split along axis 1 (columns)
result_axis1 = np.split(arr_2d, 2, axis=1)
print("\nSplit along axis 1 (columns):")
for i, subarr in enumerate(result_axis1):
    print(f"Subarray {i}:")
    print(subarr)
Output:
Original 2D array:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

Split along axis 0 (rows):
Subarray 0:
[[0 1 2 3]
 [4 5 6 7]]
Subarray 1:
[[ 8  9 10 11]
 [12 13 14 15]]

Split along axis 1 (columns):
Subarray 0:
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
Subarray 1:
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]
1.2 Horizontal Splitting with np.hsplit()
np.hsplit() is a convenience function that splits arrays horizontally (column-wise). It's equivalent to np.split(arr, indices_or_sections, axis=1).
Syntax:
pythonnp.hsplit(array, indices_or_sections)
Example:
python# Create a 2D array
arr_2d = np.arange(16).reshape(4, 4)
print("Original 2D array:")
print(arr_2d)

# Horizontal split into 2 equal parts
result = np.hsplit(arr_2d, 2)
print("\nHorizontal split into 2 equal parts:")
for i, subarr in enumerate(result):
    print(f"Subarray {i}:")
    print(subarr)

# Horizontal split at specific column indices
result_indices = np.hsplit(arr_2d, [1, 3])
print("\nHorizontal split at indices [1, 3]:")
for i, subarr in enumerate(result_indices):
    print(f"Subarray {i}:")
    print(subarr)
Output:
Original 2D array:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

Horizontal split into 2 equal parts:
Subarray 0:
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
Subarray 1:
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]

Horizontal split at indices [1, 3]:
Subarray 0:
[[ 0]
 [ 4]
 [ 8]
 [12]]
Subarray 1:
[[ 1  2]
 [ 5  6]
 [ 9 10]
 [13 14]]
Subarray 2:
[[ 3]
 [ 7]
 [11]
 [15]]
1.3 Vertical Splitting with np.vsplit()
np.vsplit() is a convenience function that splits arrays vertically (row-wise). It's equivalent to np.split(arr, indices_or_sections, axis=0).
Syntax:
pythonnp.vsplit(array, indices_or_sections)
Example:
python# Create a 2D array
arr_2d = np.arange(16).reshape(4, 4)
print("Original 2D array:")
print(arr_2d)

# Vertical split into 2 equal parts
result = np.vsplit(arr_2d, 2)
print("\nVertical split into 2 equal parts:")
for i, subarr in enumerate(result):
    print(f"Subarray {i}:")
    print(subarr)

# Vertical split at specific row indices
result_indices = np.vsplit(arr_2d, [1, 3])
print("\nVertical split at indices [1, 3]:")
for i, subarr in enumerate(result_indices):
    print(f"Subarray {i}:")
    print(subarr)
Output:
Original 2D array:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

Vertical split into 2 equal parts:
Subarray 0:
[[0 1 2 3]
 [4 5 6 7]]
Subarray 1:
[[ 8  9 10 11]
 [12 13 14 15]]

Vertical split at indices [1, 3]:
Subarray 0:
[[0 1 2 3]]
Subarray 1:
[[ 4  5  6  7]
 [ 8  9 10 11]]
Subarray 2:
[[12 13 14 15]]
1.4 Error Handling in Array Splitting
When splitting arrays, you might encounter errors if:

The array cannot be split into equal parts
The indices are out of bounds
The number of sections is not compatible with the array size

python# Example: Trying to split a length-10 array into 3 equal parts
arr = np.arange(10)

try:
    result = np.split(arr, 3)
except ValueError as e:
    print(f"Error: {e}")
Output:
Error: array split does not result in an equal division
To handle this, you can either:

Use np.array_split() which allows unequal divisions
Ensure your array size is divisible by the number of sections
Specify exact indices for splitting

python# Using array_split for unequal divisions
arr = np.arange(10)
result = np.array_split(arr, 3)
print("Split into 3 parts with array_split:")
for i, subarr in enumerate(result):
    print(f"Subarray {i}:", subarr)
Output:
Split into 3 parts with array_split:
Subarray 0: [0 1 2 3]
Subarray 1: [4 5 6]
Subarray 2: [7 8 9]
1.5 Practical Applications
Array splitting is useful in many scenarios:

Data batching for machine learning:

python# Creating mini-batches for training
data = np.random.randn(100, 4)  # 100 samples with 4 features
batch_size = 10
batches = np.array_split(data, 100 // batch_size)
print(f"Created {len(batches)} batches of shape {batches[0].shape}")

Cross-validation in machine learning:

python# 5-fold cross-validation split
data = np.random.randn(100, 4)
k_folds = 5
folds = np.array_split(data, k_folds)
print(f"Created {len(folds)} folds of approximately {folds[0].shape[0]} samples each")

Image processing:

python# Splitting an image into tiles
image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)  # RGB image
tiles_v = np.vsplit(image, 3)  # Split into 3 rows
tiles = [np.hsplit(row, 3) for row in tiles_v]  # Split each row into 3 columns
print(f"Split image into 9 tiles of shape {tiles[0][0].shape}")
1.6 Module 1 Exercises
Exercise 1: Write a function that takes a 2D array and splits it into 4 equal quadrants.
Exercise 2: Create a function that performs a sliding window split on a 1D array with a specified window size and stride.
Exercise 3: Split an array of student test scores into quartiles (bottom 25%, 25-50%, 50-75%, top 25%).
Module 2: Linear Algebra with NumPy
Introduction
Linear algebra operations are fundamental to many scientific and engineering applications. NumPy's linear algebra module (np.linalg) provides efficient implementations of these operations, including calculating determinants, eigenvalues, and eigenvectors.
2.1 Determinants with np.linalg.det()
The determinant is a scalar value calculated from a square matrix and has important geometric and algebraic interpretations.
Syntax:
pythonnp.linalg.det(A)
Where A is a square matrix.
Properties of determinants:

If the determinant is zero, the matrix is singular (non-invertible)
The determinant of a product of matrices equals the product of their determinants
The determinant of a transposed matrix equals the determinant of the original matrix

Example 1: Basic determinant calculation
pythonimport numpy as np

# Create a 2x2 matrix
A = np.array([[4, 3], 
              [2, 1]])
det_A = np.linalg.det(A)
print(f"Matrix A:\n{A}")
print(f"Determinant of A: {det_A}")

# Create a 3x3 matrix
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]])
det_B = np.linalg.det(B)
print(f"\nMatrix B:\n{B}")
print(f"Determinant of B: {det_B}")
Output:
Matrix A:
[[4 3]
 [2 1]]
Determinant of A: -2.0

Matrix B:
[[1 2 3]
 [4 5 6]
 [7 8 0]]
Determinant of B: 27.0
Example 2: Special matrices
python# Identity matrix - determinant should be 1
I = np.eye(3)
print(f"Identity matrix:\n{I}")
print(f"Determinant of identity matrix: {np.linalg.det(I)}")

# Singular matrix - determinant should be 0
S = np.array([[1, 2], 
              [2, 4]])  # Second row is 2x first row
print(f"\nSingular matrix:\n{S}")
print(f"Determinant of singular matrix: {np.linalg.det(S)}")

# Rotation matrix (90 degrees) - determinant should be 1
theta = np.pi/2  # 90 degrees
R = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]])
print(f"\nRotation matrix:\n{R}")
print(f"Determinant of rotation matrix: {np.linalg.det(R)}")
Output:
Identity matrix:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Determinant of identity matrix: 1.0

Singular matrix:
[[1 2]
 [2 4]]
Determinant of singular matrix: 0.0

Rotation matrix:
[[ 6.123234e-17 -1.000000e+00]
 [ 1.000000e+00  6.123234e-17]]
Determinant of rotation matrix: 1.0
Example 3: Determinant applications
python# Area of a parallelogram
vectors = np.array([[3, 1], 
                    [2, 4]])
area = abs(np.linalg.det(vectors))
print(f"Vectors defining a parallelogram:\n{vectors}")
print(f"Area of the parallelogram: {area}")

# Volume of a parallelepiped
vectors_3d = np.array([[1, 0, 0], 
                       [0, 2, 0], 
                       [0, 0, 3]])
volume = abs(np.linalg.det(vectors_3d))
print(f"\nVectors defining a parallelepiped:\n{vectors_3d}")
print(f"Volume of the parallelepiped: {volume}")

# Check if points are collinear
p1 = np.array([1, 1])
p2 = np.array([3, 3])
p3 = np.array([5, 5])
matrix = np.column_stack([p1, p2, p3, np.ones(3)])
are_collinear = abs(np.linalg.det(matrix[:3, :3])) < 1e-10
print(f"\nPoints: {p1}, {p2}, {p3}")
print(f"Are the points collinear? {are_collinear}")
Output:
Vectors defining a parallelogram:
[[3 1]
 [2 4]]
Area of the parallelogram: 10.0

Vectors defining a parallelepiped:
[[1 0 0]
 [0 2 0]
 [0 0 3]]
Volume of the parallelepiped: 6.0

Points: [1 1], [3 3], [5 5]
Are the points collinear? True
2.2 Eigenvalues and Eigenvectors with np.linalg.eig()
Eigenvalues and eigenvectors are fundamental concepts in linear algebra with wide applications in physics, engineering, data science, and machine learning.
Definition: If A is a square matrix, then a non-zero vector v is an eigenvector of A if there exists a scalar λ (lambda) such that:
A·v = λ·v
Here, λ is called the eigenvalue corresponding to the eigenvector v.
Syntax:
pythoneigenvalues, eigenvectors = np.linalg.eig(A)
Where:

A: A square matrix
eigenvalues: A 1D array containing the eigenvalues
eigenvectors: A 2D array where each column corresponds to an eigenvector

Example 1: Basic eigenvalue and eigenvector calculation
pythonimport numpy as np

# Create a 2x2 matrix
A = np.array([[4, -2], 
              [1, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Matrix A:\n{A}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors (as columns):\n{eigenvectors}")

# Verify that A·v = λ·v for the first eigenvalue-eigenvector pair
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0]
Av1 = np.dot(A, v1)
lambda_v1 = lambda1 * v1

print(f"\nVerification for first eigenpair:")
print(f"A·v1 = {Av1}")
print(f"λ1·v1 = {lambda_v1}")
print(f"Are they equal? {np.allclose(Av1, lambda_v1)}")
Output:
Matrix A:
[[ 4 -2]
 [ 1  1]]

Eigenvalues: [3.+0.j 2.+0.j]

Eigenvectors (as columns):
[[ 0.89442719  0.70710678]
 [ 0.4472136   0.70710678]]

Verification for first eigenpair:
A·v1 = [2.68328157+0.j 1.3416408 +0.j]
λ1·v1 = [2.68328157+0.j 1.3416408 +0.j]
Are they equal? True
Example 2: Symmetric matrices
Symmetric matrices have special properties regarding eigenvalues and eigenvectors:

All eigenvalues are real
Eigenvectors corresponding to distinct eigenvalues are orthogonal

python# Create a symmetric matrix
S = np.array([[2, 1], 
              [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(S)

print(f"Symmetric matrix S:\n{S}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors (as columns):\n{eigenvectors}")

# Check orthogonality of eigenvectors
dot_product = np.dot(eigenvectors[:, 0], eigenvectors[:, 1])
print(f"\nDot product of eigenvectors: {dot_product}")
print(f"Are the eigenvectors orthogonal? {abs(dot_product) < 1e-10}")
Output:
Symmetric matrix S:
[[2 1]
 [1 2]]

Eigenvalues: [3.+0.j 1.+0.j]

Eigenvectors (as columns):
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]

Dot product of eigenvectors: 0.0
Are the eigenvectors orthogonal? True
Example 3: Applications of eigenvalues and eigenvectors
python# Principal Component Analysis (PCA) - simplified example
# Create a covariance matrix
cov_matrix = np.array([[4, 2], 
                       [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("PCA Example:")
print(f"Covariance matrix:\n{cov_matrix}")
print(f"\nEigenvalues (variances along principal components): {eigenvalues}")
print(f"\nEigenvectors (principal components as columns):\n{eigenvectors}")
print(f"\nExplained variance ratio: {eigenvalues / np.sum(eigenvalues)}")

# Dynamical systems - population growth model
# Matrix representing population transitions
transition = np.array([[0.8, 0.3], 
                       [0.2, 0.7]])
eigenvalues, eigenvectors = np.linalg.eig(transition)

print("\nDynamical System Example:")
print(f"Population transition matrix:\n{transition}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nDominant eigenvalue (long-term growth factor): {max(abs(eigenvalues))}")
Output:
PCA Example:
Covariance matrix:
[[4 2]
 [2 3]]

Eigenvalues (variances along principal components): [5.+0.j 2.+0.j]

Eigenvectors (principal components as columns):
[[ 0.83205029  0.5547002 ]
 [ 0.5547002  -0.83205029]]

Explained variance ratio: [0.71428571+0.j 0.28571429+0.j]

Dynamical System Example:
Population transition matrix:
[[0.8 0.3]
 [0.2 0.7]]

Eigenvalues: [1.+0.j 0.5+0.j]

Dominant eigenvalue (long-term growth factor): 1.0
2.3 Complex Eigenvalues and Eigenvectors
Some matrices have complex eigenvalues and eigenvectors, especially matrices representing rotations or oscillations.
python# Create a rotation matrix (counterclockwise rotation by 90 degrees)
theta = np.pi/2
R = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]])
eigenvalues, eigenvectors = np.linalg.eig(R)

print("Rotation Matrix Analysis:")
print(f"Rotation matrix:\n{R}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")

# Visualize complex eigenvalues
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(eigenvalues.real, eigenvalues.imag, color='red', s=100)
circle = plt.Circle((0, 0), 1, fill=False, color='blue', linestyle='--')
plt.gca().add_patch(circle)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Eigenvalues in Complex Plane')
plt.axis('equal')
Note: The eigenvalues of a rotation matrix have magnitude 1 and lie on the unit circle in the complex plane.
2.4 Linear Algebra Applications
1. Solving Systems of Linear Equations
python# Solve Ax = b using eigendecomposition
A = np.array([[4, 2], 
              [1, 3]])
b = np.array([8, 7])

# Get eigendecomposition A = PDP^-1
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

# Solve in eigenspace: x = P D^-1 P^-1 b
D_inv = np.diag(1/eigenvalues)
x = P @ D_inv @ P_inv @ b

print("Solving System of Linear Equations:")
print(f"Matrix A:\n{A}")
print(f"Vector b: {b}")
print(f"Solution x: {x}")
print(f"Verification A·x: {A @ x}")
print(f"Are A·x and b equal? {np.allclose(A @ x, b)}")
2. Matrix Diagonalization
python# Diagonalize a matrix
A = np.array([[2, 1], 
              [1, 2]])
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

print("\nMatrix Diagonalization:")
print(f"Original matrix A:\n{A}")
print(f"Diagonal matrix D:\n{D}")
print(f"Eigenvector matrix P:\n{P}")
print(f"Reconstructed A = PDP^-1:\n{A_reconstructed}")
print(f"Is reconstruction accurate? {np.allclose(A, A_reconstructed)}")
3. Computing Matrix Powers Efficiently
python# Compute A^10 efficiently using eigendecomposition
A = np.array([[2, 1], 
              [1, 2]])
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

# A^n = P D^n P^-1
n = 10
D_n = np.diag(eigenvalues**n)
A_n = P @ D_n @ P_inv

# Verify with direct multiplication
A_n_direct = np.linalg.matrix_power(A, n)

print("\nEfficient Matrix Powers:")
print(f"A^10 via eigendecomposition:\n{A_n}")
print(f"A^10 via direct multiplication:\n{A_n_direct}")
print(f"Are they equal? {np.allclose(A_n, A_n_direct)}")
2.5 Module 2 Exercises
Exercise 1: Write a function that determines if a matrix is diagonalizable based on its eigenvalues and eigenvectors.
Exercise 2: Create a function that computes the condition number of a matrix using its eigenvalues.
Exercise 3: Implement a Markov chain simulation using eigenvalues to find the steady-state distribution.
Exercise 4: Use eigendecomposition to implement PCA on a 2D dataset and visualize the principal components.
Module 3: Statistical Analysis with NumPy
Introduction
NumPy provides essential functions for statistical analysis, including descriptive statistics, correlation, and covariance calculations. These tools are fundamental for data exploration, analysis, and building statistical models.
3.1 Descriptive Statistics
Descriptive statistics summarize and describe the main features of a dataset. NumPy provides functions for calculating common statistical measures:
3.1.1 Mean with np.mean()
The mean (average) is the sum of all values divided by the number of values.
Syntax:
pythonnp.mean(arr, axis=None, dtype=None, keepdims=False)
Where:

arr: Input array
axis: Axis along which to compute the mean (default: compute over the flattened array)
dtype: Type to use in computing the mean
keepdims: If True, the reduced dimensions are kept as size 1

Example:
pythonimport numpy as np

# Create sample data
data = np.array([4, 8, 6, 5, 7])
print(f"Data: {data}")

# Calculate the mean
mean_value = np.mean(data)
print(f"Mean: {mean_value}")

# 2D array example
data_2d = np.array([[1, 2, 3], 
                    [4, 5, 6], 
                    [7, 8, 9]])
print(f"\n2D data:\n{data_2d}")

# Mean of entire array
print(f"Mean of entire array: {np.mean(data_2d)}")

# Mean along axis 0 (column means)
print(f"Column means: {np.mean(data_2d, axis=0)}")

# Mean along axis 1 (row means)
print(f"Row means: {np.mean(data_2d, axis=1)}")
Output:
Data: [4 8 6 5 7]
Mean: 6.0

2D data:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Mean of entire array: 5.0
Column means: [4. 5. 6.]
Row means: [2. 5. 8.]
3.1.2 Median with np.median()
The median is the middle value when data is arranged in order. It's less sensitive to outliers than the mean.
Syntax:
pythonnp.median(arr, axis=None, keepdims=False)
Example:
python# Create sample data with an outlier
data = np.array([4, 8, 6, 5, 7, 100])
print(f"Data with outlier: {data}")

# Calculate mean and median
mean_value = np.mean(data)
median_value = np.median(data)
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")

# 2D array example
data_2d = np.array([[1, 2, 30], 
                    [4, 5, 6], 
                    [7, 8, 9]])
print(f"\n2D data with outlier:\n{data_2d}")

# Median along axis 0 (column medians)
print(f"Column medians: {np.median(
