"""Test suite for Project 05"""
import pytest
from exercise import count_words, find_unique_elements, set_union, set_intersection, set_difference

def test_count_words():
    assert count_words("hello world hello") == {"hello": 2, "world": 1}

def test_find_unique_elements():
    assert sorted(find_unique_elements([1, 2, 2, 3, 3, 3])) == [1, 2, 3]

def test_set_union():
    assert set_union({1, 2}, {2, 3}) == {1, 2, 3}

def test_set_intersection():
    assert set_intersection({1, 2}, {2, 3}) == {2}

def test_set_difference():
    assert set_difference({1, 2}, {2, 3}) == {1}

