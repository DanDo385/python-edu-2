"""Test suite for Project 04: Lists and Tuples"""
import pytest
from exercise import (
    find_minimum, find_maximum, calculate_average,
    merge_sorted_lists, reverse_list, remove_duplicates,
    list_comprehension_squares, get_even_numbers, tuple_operations
)

def test_find_minimum():
    assert find_minimum([3, 1, 4, 1, 5]) == 1
    assert find_minimum([-5, -2, -10]) == -10
    assert find_minimum([]) is None

def test_find_maximum():
    assert find_maximum([3, 1, 4, 1, 5]) == 5
    assert find_maximum([-5, -2, -10]) == -2

def test_calculate_average():
    assert abs(calculate_average([1, 2, 3, 4, 5]) - 3.0) < 0.001
    assert abs(calculate_average([10, 20, 30]) - 20.0) < 0.001

def test_merge_sorted_lists():
    assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted_lists([1, 2], [3, 4]) == [1, 2, 3, 4]

def test_reverse_list():
    lst = [1, 2, 3, 4]
    reverse_list(lst)
    assert lst == [4, 3, 2, 1]

def test_remove_duplicates():
    assert remove_duplicates([1, 2, 2, 3, 3, 3]) == [1, 2, 3]
    assert remove_duplicates([1, 1, 1]) == [1]

def test_list_comprehension_squares():
    assert list_comprehension_squares(5) == [0, 1, 4, 9, 16]

def test_get_even_numbers():
    assert get_even_numbers([1, 2, 3, 4, 5, 6]) == [2, 4, 6]

def test_tuple_operations():
    sum_result, product_result = tuple_operations()
    assert sum_result == 7
    assert product_result == 12

