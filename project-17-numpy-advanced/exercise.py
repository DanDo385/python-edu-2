"""
Project 17: Advanced NumPy – Broadcasting and Matrix Operations

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""

import numpy as np


def add_vector_to_matrix_rows(matrix, vector):
    """
    Add a vector to each row of a matrix using broadcasting.
    
    This is a common operation in ML: adding a bias vector to each sample.
    
    Args:
        matrix (np.ndarray): Matrix of shape (m, n)
        vector (np.ndarray): Vector of shape (n,)
    
    Returns:
        np.ndarray: Matrix with vector added to each row, shape (m, n)
    """
    # TODO: Use broadcasting to add vector to each row
    # Hint: NumPy will automatically broadcast vector to match matrix
    return None


def matrix_multiply(A, B):
    """
    Multiply two matrices.
    
    Matrix multiplication: C = A @ B
    A must have shape (m, n), B must have shape (n, p)
    Result has shape (m, p)
    
    Args:
        A (np.ndarray): First matrix, shape (m, n)
        B (np.ndarray): Second matrix, shape (n, p)
    
    Returns:
        np.ndarray: Matrix product, shape (m, p)
    """
    # TODO: Perform matrix multiplication
    # Hint: Use @ operator or np.dot()
    return None


def transpose_matrix(matrix):
    """
    Transpose a matrix (swap rows and columns).
    
    Args:
        matrix (np.ndarray): Input matrix
    
    Returns:
        np.ndarray: Transposed matrix
    """
    # TODO: Transpose matrix
    # Hint: Use .T attribute or np.transpose()
    return None


def compute_euclidean_distances(points1, points2):
    """
    Compute Euclidean distances between each pair of points using broadcasting.
    
    points1: (m, d) - m points with d dimensions
    points2: (n, d) - n points with d dimensions
    Result: (m, n) - distance matrix
    
    Formula: sqrt(sum((p1 - p2)^2))
    
    Args:
        points1 (np.ndarray): First set of points, shape (m, d)
        points2 (np.ndarray): Second set of points, shape (n, d)
    
    Returns:
        np.ndarray: Distance matrix, shape (m, n)
    """
    # TODO: Compute distances using broadcasting
    # Hint: Use broadcasting to compute differences, then sum and sqrt
    # Strategy:
    # 1. Reshape points1 to (m, 1, d) and points2 to (1, n, d)
    # 2. Compute differences (broadcasting will create (m, n, d))
    # 3. Square differences, sum along last axis, take sqrt
    return None


def apply_operation_along_axis(arr, operation='sum', axis=0):
    """
    Apply operation along a specific axis.
    
    Args:
        arr (np.ndarray): Input array
        operation (str): 'sum', 'mean', 'max', or 'min'
        axis (int): Axis to operate along (0=rows, 1=columns)
    
    Returns:
        np.ndarray: Result of operation along axis
    """
    # TODO: Apply operation along specified axis
    # Hint: Use np.sum(), np.mean(), np.max(), or np.min() with axis parameter
    return None


def normalize_matrix_rows(matrix):
    """
    Normalize each row of a matrix to have unit length (L2 norm = 1).
    
    For each row, divide by its L2 norm (Euclidean length).
    
    Args:
        matrix (np.ndarray): Input matrix, shape (m, n)
    
    Returns:
        np.ndarray: Normalized matrix, same shape
    """
    # TODO: Normalize each row
    # Hint: Compute L2 norm for each row, then divide
    # L2 norm = sqrt(sum(x^2)) for each row
    return None


def outer_product(vec1, vec2):
    """
    Compute outer product of two vectors.
    
    Outer product: result[i, j] = vec1[i] * vec2[j]
    
    Args:
        vec1 (np.ndarray): First vector, shape (m,)
        vec2 (np.ndarray): Second vector, shape (n,)
    
    Returns:
        np.ndarray: Outer product matrix, shape (m, n)
    """
    # TODO: Compute outer product using broadcasting
    # Hint: Reshape vec1 to (m, 1) and vec2 to (1, n), then multiply
    return None

