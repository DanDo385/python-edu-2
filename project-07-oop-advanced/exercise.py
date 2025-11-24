"""
Project 07: Python OOP Advanced (Inheritance & Exceptions)

Complete the class definitions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""


class Shape:
    """
    Base class for all shapes.
    
    This is an abstract base class that defines the interface
    that all shapes must implement.
    """
    
    def area(self):
        """
        Calculate the area of the shape.
        
        This method should be overridden by subclasses.
        
        Returns:
            float: Area of the shape
        """
        # TODO: Raise NotImplementedError with message "Subclass must implement area()"
        pass
    
    def perimeter(self):
        """
        Calculate the perimeter of the shape.
        
        This method should be overridden by subclasses.
        
        Returns:
            float: Perimeter of the shape
        """
        # TODO: Raise NotImplementedError with message "Subclass must implement perimeter()"
        pass


class Rectangle(Shape):
    """
    A rectangle that inherits from Shape.
    
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
        
        Overrides the parent class method.
        
        Returns:
            float: Area (width * height)
        """
        # TODO: Return width * height
        return None
    
    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.
        
        Overrides the parent class method.
        
        Returns:
            float: Perimeter (2 * (width + height))
        """
        # TODO: Return 2 * (width + height)
        return None


class Circle(Shape):
    """
    A circle that inherits from Shape.
    
    Attributes:
        radius (float): Radius of circle
    """
    
    def __init__(self, radius):
        """
        Initialize a circle.
        
        Args:
            radius (float): Radius
        """
        # TODO: Set self.radius
        pass
    
    def area(self):
        """
        Calculate the area of the circle.
        
        Formula: π * radius²
        
        Returns:
            float: Area of circle
        """
        # TODO: Return π * radius²
        # Hint: Use math.pi
        import math
        return None
    
    def perimeter(self):
        """
        Calculate the perimeter (circumference) of the circle.
        
        Formula: 2 * π * radius
        
        Returns:
            float: Circumference of circle
        """
        # TODO: Return 2 * π * radius
        import math
        return None


class OverdrawError(Exception):
    """
    Custom exception for account overdraw attempts.
    
    This exception should be raised when trying to withdraw
    more money than is available in an account.
    """
    pass


class SafeBankAccount:
    """
    A bank account with exception handling.
    
    Attributes:
        balance (float): Current account balance
    """
    
    def __init__(self, initial_balance=0.0):
        """
        Initialize a bank account.
        
        Args:
            initial_balance (float): Starting balance
        """
        # TODO: Set self.balance
        pass
    
    def withdraw(self, amount):
        """
        Withdraw money from the account.
        
        Raises OverdrawError if withdrawal amount exceeds balance.
        
        Args:
            amount (float): Amount to withdraw
        
        Raises:
            OverdrawError: If amount > balance
        """
        # TODO: Check if amount > self.balance
        # TODO: If so, raise OverdrawError with message "Insufficient funds"
        # TODO: Otherwise, subtract amount from balance
        pass
    
    def get_balance(self):
        """
        Get the current balance.
        
        Returns:
            float: Current balance
        """
        # TODO: Return self.balance
        return None


def divide_numbers(a, b):
    """
    Divide two numbers with exception handling.
    
    Args:
        a (float): Dividend
        b (float): Divisor
    
    Returns:
        float: Result of division, or None if division by zero
    
    Raises:
        ZeroDivisionError: If b is 0 (handled internally)
    """
    # TODO: Use try/except to handle ZeroDivisionError
    # TODO: Return None if division by zero occurs
    return None

