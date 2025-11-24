"""
Project 17: Advanced NumPy – Broadcasting and Matrix Operations - SOLUTION

Complete solution with detailed comments explaining broadcasting, matrix operations,
and their critical role in machine learning.
"""

import numpy as np


def add_vector_to_matrix_rows(matrix, vector):
    """
    Add a vector to each row of a matrix using broadcasting.
    
    This demonstrates one of the most important broadcasting patterns in ML.
    When you have a batch of samples (matrix) and want to add a bias vector
    to each sample, broadcasting does this automatically without loops.
    
    Broadcasting rules:
    1. Align dimensions from the right
    2. Dimensions of size 1 can be "stretched"
    3. Missing dimensions are treated as size 1
    
    In this case:
    - matrix: (m, n) - m rows (samples), n columns (features)
    - vector: (n,) - n elements
    - NumPy treats vector as (1, n) and broadcasts to (m, n)
    
    Time complexity: O(m*n) but vectorized (much faster than Python loop)
    Space complexity: O(m*n) for result
    
    Args:
        matrix (np.ndarray): Matrix of shape (m, n)
        vector (np.ndarray): Vector of shape (n,)
    
    Returns:
        np.ndarray: Matrix with vector added to each row, shape (m, n)
    
    Example:
        matrix = [[1, 2], [3, 4]]  # Shape (2, 2)
        vector = [10, 20]          # Shape (2,)
        Result: [[11, 22], [13, 24]]  # Vector added to each row
    """
    # Broadcasting happens automatically!
    # NumPy sees:
    # - matrix shape: (m, n)
    # - vector shape: (n,) which it treats as (1, n)
    # - Broadcasts vector to (m, n) by repeating it m times
    # - Then adds element-wise
    
    # This is exactly how bias terms are added in neural networks:
    # output = input @ weights + bias
    # Where input is (batch_size, features) and bias is (features,)
    return matrix + vector


def matrix_multiply(A, B):
    """
    Multiply two matrices.
    
    Matrix multiplication (also called matrix product or dot product) is the
    fundamental operation in neural networks. Every layer performs:
    output = input @ weights
    
    Rules:
    - A must have shape (m, n)
    - B must have shape (n, p)
    - Result has shape (m, p)
    - Each element (i, j) = sum of A[i, :] * B[:, j]
    
    This is NOT element-wise multiplication (*), it's matrix multiplication (@).
    
    Time complexity: O(m*n*p) - but highly optimized in NumPy
    Space complexity: O(m*p) for result
    
    Args:
        A (np.ndarray): First matrix, shape (m, n)
        B (np.ndarray): Second matrix, shape (n, p)
    
    Returns:
        np.ndarray: Matrix product, shape (m, p)
    
    Example:
        A = [[1, 2],      B = [[5, 6],
             [3, 4]]          [7, 8]]
        
        Result = [[1*5+2*7, 1*6+2*8],   = [[19, 22],
                  [3*5+4*7, 3*6+4*8]]      [43, 50]]
    """
    # Use @ operator for matrix multiplication (Python 3.5+)
    # This is the same as np.dot(A, B) but more readable
    # In neural networks, this is how we compute:
    # - Forward pass: input @ weights
    # - Backward pass: gradients flow through matrix multiplications
    return A @ B
    
    # Alternative: np.dot(A, B) also works
    # But @ is preferred as it's more readable and matches mathematical notation


def transpose_matrix(matrix):
    """
    Transpose a matrix (swap rows and columns).
    
    Transposition is crucial in linear algebra and ML. In neural networks:
    - Forward pass: input @ weights
    - Backward pass: gradients @ weights.T (transpose needed!)
    
    Transpose converts (m, n) → (n, m)
    Element at (i, j) moves to (j, i)
    
    Args:
        matrix (np.ndarray): Input matrix
    
    Returns:
        np.ndarray: Transposed matrix
    
    Example:
        [[1, 2, 3],     Transpose →    [[1, 4],
         [4, 5, 6]]                    [2, 5],
                                       [3, 6]]
    """
    # .T is a convenient attribute for transpose
    # This doesn't copy data (usually), just changes the view
    # For actual copy: matrix.T.copy()
    return matrix.T
    
    # Alternative: np.transpose(matrix) also works
    # But .T is more concise and commonly used


def compute_euclidean_distances(points1, points2):
    """
    Compute Euclidean distances between each pair of points using broadcasting.
    
    This demonstrates advanced broadcasting: computing distances between
    all pairs of points from two sets efficiently.
    
    Strategy:
    1. Reshape to enable broadcasting: (m, 1, d) and (1, n, d)
    2. Broadcasting creates (m, n, d) - all pairwise differences
    3. Square differences, sum along last axis, take sqrt
    
    This is used in:
    - K-means clustering (distance to centroids)
    - K-nearest neighbors
    - Similarity computations
    
    Args:
        points1 (np.ndarray): First set of points, shape (m, d)
        points2 (np.ndarray): Second set of points, shape (n, d)
    
    Returns:
        np.ndarray: Distance matrix, shape (m, n)
                    result[i, j] = distance between points1[i] and points2[j]
    
    Example:
        points1 = [[0, 0], [1, 1]]      # 2 points in 2D
        points2 = [[0, 0], [1, 0]]      # 2 points in 2D
        Result: [[0, 1], [√2, 1]]       # 2x2 distance matrix
    """
    # Reshape points1 to (m, 1, d) - adds dimension for broadcasting
    # This allows broadcasting with points2
    points1_expanded = points1[:, np.newaxis, :]  # Shape: (m, 1, d)
    
    # Reshape points2 to (1, n, d) - adds dimension for broadcasting
    points2_expanded = points2[np.newaxis, :, :]  # Shape: (1, n, d)
    
    # Broadcasting creates (m, n, d) - all pairwise differences
    # Each element [i, j, :] is points1[i] - points2[j]
    differences = points1_expanded - points2_expanded
    
    # Square the differences (element-wise)
    squared_differences = differences ** 2
    
    # Sum along the last axis (dimension d) to get squared distances
    # Result shape: (m, n) - one distance per pair
    squared_distances = np.sum(squared_differences, axis=2)
    
    # Take square root to get actual Euclidean distances
    distances = np.sqrt(squared_distances)
    
    return distances


def apply_operation_along_axis(arr, operation='sum', axis=0):
    """
    Apply operation along a specific axis.
    
    Understanding axes is crucial in ML:
    - axis=0: operate along rows (down the columns)
    - axis=1: operate along columns (across the rows)
    - axis=-1: operate along last dimension
    
    This is used for:
    - Batch normalization (normalize across features)
    - Aggregating predictions (sum/mean across samples)
    - Computing statistics along specific dimensions
    
    Args:
        arr (np.ndarray): Input array
        operation (str): 'sum', 'mean', 'max', or 'min'
        axis (int): Axis to operate along (0=rows, 1=columns)
    
    Returns:
        np.ndarray: Result of operation along axis
    
    Example:
        arr = [[1, 2, 3],
               [4, 5, 6]]
        sum along axis=0: [5, 7, 9]  (sum down columns)
        sum along axis=1: [6, 15]    (sum across rows)
    """
    # Apply the requested operation along the specified axis
    if operation == 'sum':
        # Sum along axis - reduces that dimension
        return np.sum(arr, axis=axis)
    
    elif operation == 'mean':
        # Mean along axis - average across that dimension
        return np.mean(arr, axis=axis)
    
    elif operation == 'max':
        # Maximum along axis - find max across that dimension
        return np.max(arr, axis=axis)
    
    elif operation == 'min':
        # Minimum along axis - find min across that dimension
        return np.min(arr, axis=axis)
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def normalize_matrix_rows(matrix):
    """
    Normalize each row of a matrix to have unit length (L2 norm = 1).
    
    L2 normalization scales each row so its Euclidean length is 1.
    This is used in:
    - Feature normalization
    - Cosine similarity computations
    - Making vectors comparable regardless of magnitude
    
    Formula: normalized_row = row / ||row||_2
    Where ||row||_2 = sqrt(sum(row^2))
    
    Args:
        matrix (np.ndarray): Input matrix, shape (m, n)
    
    Returns:
        np.ndarray: Normalized matrix, same shape
                    Each row has L2 norm = 1
    
    Example:
        [[3, 4]] → [[0.6, 0.8]]  (3^2 + 4^2 = 25, sqrt = 5, so 3/5=0.6, 4/5=0.8)
    """
    # Compute L2 norm for each row
    # np.linalg.norm(matrix, axis=1) computes sqrt(sum(x^2)) for each row
    # Result shape: (m,) - one norm per row
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # keepdims=True keeps shape as (m, 1) instead of (m,)
    # This enables broadcasting when dividing
    
    # Divide each row by its norm
    # Broadcasting: (m, n) / (m, 1) → (m, n)
    # Each row is divided by its corresponding norm
    normalized = matrix / norms
    
    return normalized


def outer_product(vec1, vec2):
    """
    Compute outer product of two vectors.
    
    Outer product creates a matrix from two vectors.
    result[i, j] = vec1[i] * vec2[j]
    
    This is different from:
    - Inner product (dot product): vec1 @ vec2 → scalar
    - Element-wise product: vec1 * vec2 → vector (if same length)
    
    Used in:
    - Computing covariance matrices
    - Outer product layers in neural networks
    - Creating rank-1 matrices
    
    Args:
        vec1 (np.ndarray): First vector, shape (m,)
        vec2 (np.ndarray): Second vector, shape (n,)
    
    Returns:
        np.ndarray: Outer product matrix, shape (m, n)
    
    Example:
        vec1 = [1, 2]    vec2 = [3, 4, 5]
        Result: [[3, 4, 5],
                 [6, 8, 10]]
    """
    # Reshape vec1 to (m, 1) - column vector
    # This enables broadcasting with vec2
    vec1_col = vec1[:, np.newaxis]  # Shape: (m, 1)
    
    # vec2 is already (n,), broadcasting treats it as (1, n)
    # When we multiply: (m, 1) * (1, n) → (m, n) via broadcasting
    # Each element [i, j] = vec1[i] * vec2[j]
    result = vec1_col * vec2
    
    return result
    
    # Alternative: np.outer(vec1, vec2) also works
    # But understanding broadcasting helps with more complex operations

