"""Test suite for Project 15: Dynamic Programming"""
import pytest
from exercise import fibonacci_memoization, fibonacci_tabulation, coin_change

def test_fibonacci_memoization():
    assert fibonacci_memoization(0) == 0
    assert fibonacci_memoization(1) == 1
    assert fibonacci_memoization(10) == 55

def test_fibonacci_tabulation():
    assert fibonacci_tabulation(0) == 0
    assert fibonacci_tabulation(1) == 1
    assert fibonacci_tabulation(10) == 55

def test_coin_change():
    assert coin_change([1, 3, 4], 6) == 2  # 3 + 3
    assert coin_change([1, 2, 5], 11) == 3  # 5 + 5 + 1
    assert coin_change([2], 3) == -1  # Impossible

