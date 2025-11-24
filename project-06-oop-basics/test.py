"""
Test suite for Project 06: Python OOP Basics

Run with: pytest test.py -v
"""

import pytest
from exercise import BankAccount, Point, Rectangle, Student


class TestBankAccount:
    """Test BankAccount class."""
    
    def test_initial_balance(self):
        account = BankAccount(100.0)
        assert account.get_balance() == 100.0
    
    def test_default_balance(self):
        account = BankAccount()
        assert account.get_balance() == 0.0
    
    def test_deposit(self):
        account = BankAccount(50.0)
        account.deposit(25.0)
        assert account.get_balance() == 75.0
    
    def test_withdraw(self):
        account = BankAccount(100.0)
        account.withdraw(30.0)
        assert account.get_balance() == 70.0
    
    def test_multiple_operations(self):
        account = BankAccount(100.0)
        account.deposit(50.0)
        account.withdraw(25.0)
        account.deposit(10.0)
        assert account.get_balance() == 135.0


class TestPoint:
    """Test Point class."""
    
    def test_distance_from_origin(self):
        point = Point(3.0, 4.0)
        assert abs(point.distance_from_origin() - 5.0) < 0.001
    
    def test_distance_to_same_point(self):
        point1 = Point(0.0, 0.0)
        point2 = Point(0.0, 0.0)
        assert abs(point1.distance_to(point2) - 0.0) < 0.001
    
    def test_distance_to_different_point(self):
        point1 = Point(0.0, 0.0)
        point2 = Point(3.0, 4.0)
        assert abs(point1.distance_to(point2) - 5.0) < 0.001
    
    def test_distance_both_ways(self):
        point1 = Point(1.0, 2.0)
        point2 = Point(4.0, 6.0)
        dist1 = point1.distance_to(point2)
        dist2 = point2.distance_to(point1)
        assert abs(dist1 - dist2) < 0.001


class TestRectangle:
    """Test Rectangle class."""
    
    def test_area(self):
        rect = Rectangle(5.0, 3.0)
        assert rect.area() == 15.0
    
    def test_perimeter(self):
        rect = Rectangle(5.0, 3.0)
        assert rect.perimeter() == 16.0
    
    def test_square(self):
        rect = Rectangle(4.0, 4.0)
        assert rect.area() == 16.0
        assert rect.perimeter() == 16.0


class TestStudent:
    """Test Student class."""
    
    def test_add_grade(self):
        student = Student("Alice")
        student.add_grade(85.0)
        student.add_grade(90.0)
        assert len(student.grades) == 2
    
    def test_get_average(self):
        student = Student("Bob")
        student.add_grade(80.0)
        student.add_grade(90.0)
        student.add_grade(100.0)
        assert abs(student.get_average() - 90.0) < 0.001
    
    def test_get_average_no_grades(self):
        student = Student("Charlie")
        assert student.get_average() == 0.0
    
    def test_multiple_students(self):
        student1 = Student("Alice")
        student2 = Student("Bob")
        student1.add_grade(100.0)
        student2.add_grade(50.0)
        assert student1.get_average() == 100.0
        assert student2.get_average() == 50.0

