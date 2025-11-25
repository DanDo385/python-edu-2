"""Test suite for Project 21: Logistic Regression"""
import pytest
import numpy as np
from exercise import (
    sigmoid,
    predict_proba,
    predict,
    binary_cross_entropy_loss,
    compute_gradients_logistic,
    train_logistic_regression,
)


def test_sigmoid():
    assert abs(sigmoid(0) - 0.5) < 0.001
    assert sigmoid(10) > 0.99
    assert sigmoid(-10) < 0.01


def test_predict_proba():
    x = np.array([0, 1, 2])
    w, b = 1.0, 0.0
    proba = predict_proba(x, w, b)
    assert len(proba) == 3
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_predict():
    x = np.array([-2, 0, 2])
    w, b = 1.0, 0.0
    predictions = predict(x, w, b)
    assert np.all((predictions == 0) | (predictions == 1))


def test_binary_cross_entropy_loss():
    y_pred = np.array([0.9, 0.1, 0.8])
    y_true = np.array([1, 0, 1])
    loss = binary_cross_entropy_loss(y_pred, y_true)
    assert loss > 0


def test_train_logistic_regression():
    # Create separable data
    np.random.seed(42)
    x = np.concatenate([np.random.randn(20) - 2, np.random.randn(20) + 2])
    y = np.concatenate([np.zeros(20), np.ones(20)])
    
    w, b, history = train_logistic_regression(x, y, learning_rate=0.1, epochs=500)
    assert len(history) == 500
    assert history[-1] < history[0]  # Loss should decrease




