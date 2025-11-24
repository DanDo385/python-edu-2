"""Test suite for Project 22: Activation Functions"""
import pytest
import numpy as np
from exercise import relu, relu_derivative, tanh, softmax, forward_pass_simple_network


def test_relu():
    x = np.array([-2, -1, 0, 1, 2])
    result = relu(x)
    expected = np.array([0, 0, 0, 1, 2])
    assert np.array_equal(result, expected)


def test_relu_derivative():
    x = np.array([-2, -1, 0, 1, 2])
    result = relu_derivative(x)
    expected = np.array([0, 0, 0, 1, 1])
    assert np.array_equal(result, expected)


def test_tanh():
    assert abs(tanh(0) - 0.0) < 0.001
    assert tanh(10) > 0.99
    assert tanh(-10) < -0.99


def test_softmax():
    x = np.array([1, 2, 3])
    result = softmax(x)
    # Check sums to 1
    assert abs(np.sum(result) - 1.0) < 0.001
    # Check all positive
    assert np.all(result > 0)
    # Check largest input has highest probability
    assert result[2] > result[1] > result[0]


def test_forward_pass_simple_network():
    x = np.array([1, 2])
    w1, b1, w2, b2 = 1.0, 0.0, 1.0, 0.0
    result = forward_pass_simple_network(x, w1, b1, w2, b2)
    assert len(result) == 2


