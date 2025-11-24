"""Test suite for Project 07: Python OOP Advanced"""
import pytest
from exercise import Shape, Rectangle, Circle, OverdrawError, SafeBankAccount, divide_numbers

def test_shape_abstract():
    shape = Shape()
    with pytest.raises(NotImplementedError):
        shape.area()
    with pytest.raises(NotImplementedError):
        shape.perimeter()

def test_rectangle_inheritance():
    rect = Rectangle(5.0, 3.0)
    assert isinstance(rect, Shape)
    assert rect.area() == 15.0
    assert rect.perimeter() == 16.0

def test_circle_inheritance():
    import math
    circle = Circle(3.0)
    assert isinstance(circle, Shape)
    assert abs(circle.area() - math.pi * 9) < 0.001
    assert abs(circle.perimeter() - 2 * math.pi * 3) < 0.001

def test_overdraw_error():
    account = SafeBankAccount(100.0)
    with pytest.raises(OverdrawError):
        account.withdraw(150.0)

def test_safe_withdraw():
    account = SafeBankAccount(100.0)
    account.withdraw(50.0)
    assert account.get_balance() == 50.0

def test_divide_numbers():
    assert divide_numbers(10, 2) == 5.0
    assert divide_numbers(10, 0) is None

