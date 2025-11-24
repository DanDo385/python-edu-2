"""Test suite for Project 17: Advanced NumPy"""
import pytest
import numpy as np
from exercise import (
    add_vector_to_matrix_rows,
    matrix_multiply,
    transpose_matrix,
    compute_euclidean_distances,
    apply_operation_along_axis,
    normalize_matrix_rows,
    outer_product,
)


class TestBroadcasting:
    """Test broadcasting operations."""
    
    def test_add_vector_to_matrix_rows(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        vector = np.array([10, 20, 30])
        result = add_vector_to_matrix_rows(matrix, vector)
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        assert np.allclose(result, expected)


class TestMatrixOperations:
    """Test matrix operations."""
    
    def test_matrix_multiply(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result, expected)
    
    def test_transpose_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose_matrix(matrix)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert np.array_equal(result, expected)


class TestDistances:
    """Test distance computations."""
    
    def test_euclidean_distances(self):
        points1 = np.array([[0, 0], [1, 1]])
        points2 = np.array([[0, 0], [1, 0], [0, 1]])
        result = compute_euclidean_distances(points1, points2)
        assert result.shape == (2, 3)
        assert abs(result[0, 0] - 0.0) < 0.001  # Same point


class TestAxisOperations:
    """Test operations along axes."""
    
    def test_sum_along_axis(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = apply_operation_along_axis(arr, 'sum', axis=0)
        expected = np.array([5, 7, 9])
        assert np.array_equal(result, expected)
    
    def test_mean_along_axis(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = apply_operation_along_axis(arr, 'mean', axis=1)
        expected = np.array([2.0, 5.0])
        assert np.allclose(result, expected)


class TestNormalization:
    """Test normalization operations."""
    
    def test_normalize_matrix_rows(self):
        matrix = np.array([[3, 4], [5, 12]])  # Rows have lengths 5 and 13
        result = normalize_matrix_rows(matrix)
        # Check that each row has L2 norm = 1
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)


class TestOuterProduct:
    """Test outer product."""
    
    def test_outer_product(self):
        vec1 = np.array([1, 2])
        vec2 = np.array([3, 4, 5])
        result = outer_product(vec1, vec2)
        expected = np.array([[3, 4, 5], [6, 8, 10]])
        assert np.array_equal(result, expected)

