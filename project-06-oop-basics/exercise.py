"""
Project 06: Python OOP Basics

Complete the class definitions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""


class BankAccount:
    """
    A simple bank account class.
    
    Attributes:
        balance (float): Current account balance
    """
    
    def __init__(self, initial_balance=0.0):
        """
        Initialize a bank account with an initial balance.
        
        Args:
            initial_balance (float): Starting balance (default 0.0)
        """
        # TODO: Set self.balance to initial_balance
        pass
    
    def deposit(self, amount):
        """
        Deposit money into the account.
        
        Args:
            amount (float): Amount to deposit
        """
        # TODO: Add amount to self.balance
        pass
    
    def withdraw(self, amount):
        """
        Withdraw money from the account.
        
        Args:
            amount (float): Amount to withdraw
        """
        # TODO: Subtract amount from self.balance
        pass
    
    def get_balance(self):
        """
        Get the current account balance.
        
        Returns:
            float: Current balance
        """
        # TODO: Return self.balance
        return None


class Point:
    """
    A point in 2D space.
    
    Attributes:
        x (float): X coordinate
        y (float): Y coordinate
    """
    
    def __init__(self, x, y):
        """
        Initialize a point with x and y coordinates.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        # TODO: Set self.x and self.y
        pass
    
    def distance_from_origin(self):
        """
        Calculate distance from origin (0, 0).
        
        Formula: sqrt(x² + y²)
        
        Returns:
            float: Distance from origin
        """
        # TODO: Calculate and return distance
        # Hint: Use math.sqrt and math.pow or ** operator
        import math
        return None
    
    def distance_to(self, other_point):
        """
        Calculate distance to another point.
        
        Formula: sqrt((x2-x1)² + (y2-y1)²)
        
        Args:
            other_point (Point): Another point
        
        Returns:
            float: Distance between points
        """
        # TODO: Calculate distance between self and other_point
        import math
        return None


class Rectangle:
    """
    A rectangle with width and height.
    
    Attributes:
        width (float): Width of rectangle
        height (float): Height of rectangle
    """
    
    def __init__(self, width, height):
        """
        Initialize a rectangle.
        
        Args:
            width (float): Width
            height (float): Height
        """
        # TODO: Set self.width and self.height
        pass
    
    def area(self):
        """
        Calculate the area of the rectangle.
        
        Returns:
            float: Area (width * height)
        """
        # TODO: Calculate and return area
        return None
    
    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.
        
        Returns:
            float: Perimeter (2 * (width + height))
        """
        # TODO: Calculate and return perimeter
        return None


class Student:
    """
    A student with name and grades.
    
    Attributes:
        name (str): Student's name
        grades (list): List of grade scores
    """
    
    def __init__(self, name):
        """
        Initialize a student.
        
        Args:
            name (str): Student's name
        """
        # TODO: Set self.name and initialize self.grades as empty list
        pass
    
    def add_grade(self, grade):
        """
        Add a grade to the student's record.
        
        Args:
            grade (float): Grade score to add
        """
        # TODO: Append grade to self.grades
        pass
    
    def get_average(self):
        """
        Calculate the average of all grades.
        
        Returns:
            float: Average grade, or 0.0 if no grades
        """
        # TODO: Calculate and return average
        return None

