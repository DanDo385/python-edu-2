"""
Project 07: Python OOP Advanced (Inheritance & Exceptions) - SOLUTION

Complete solution with detailed comments explaining inheritance and exception handling.
"""


class Shape:
    """
    Base class for all shapes.
    
    This demonstrates an abstract base class pattern. In Python, we don't have
    true abstract classes without using the abc module, but we can raise
    NotImplementedError to indicate that subclasses must override methods.
    
    This is a common pattern: define the interface in the parent class,
    require subclasses to implement the actual behavior.
    """
    
    def area(self):
        """
        Calculate the area of the shape.
        
        This method is meant to be overridden by subclasses. If called
        directly on a Shape instance, it raises NotImplementedError to
        indicate that subclasses must provide an implementation.
        
        Returns:
            float: Area of the shape
        
        Raises:
            NotImplementedError: Always, since this is abstract
        """
        # Raise NotImplementedError to indicate this method must be overridden
        # This is Python's way of creating an "abstract" method
        # Subclasses (Rectangle, Circle) will override this with their own implementation
        raise NotImplementedError("Subclass must implement area()")
    
    def perimeter(self):
        """
        Calculate the perimeter of the shape.
        
        Similar to area(), this is an abstract method that must be
        implemented by subclasses.
        
        Returns:
            float: Perimeter of the shape
        
        Raises:
            NotImplementedError: Always, since this is abstract
        """
        # Raise NotImplementedError - subclasses must implement
        raise NotImplementedError("Subclass must implement perimeter()")


class Rectangle(Shape):
    """
    A rectangle that inherits from Shape.
    
    This demonstrates inheritance: Rectangle IS-A Shape, so it inherits
    all methods from Shape. However, it overrides area() and perimeter()
    to provide Rectangle-specific implementations.
    
    Inheritance allows code reuse: we don't need to redefine the interface,
    just implement the specific behavior for rectangles.
    
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
        # Store rectangle-specific attributes
        # These are not in the parent Shape class - each subclass
        # can have its own unique attributes
        self.width = width
        self.height = height
    
    def area(self):
        """
        Calculate the area of the rectangle.
        
        This method OVERRIDES the parent class's area() method.
        When you call rect.area(), Python uses this implementation
        instead of the one in Shape (which would raise NotImplementedError).
        
        This is polymorphism: the same method name (area) behaves
        differently depending on the object's type.
        
        Returns:
            float: Area (width * height)
        """
        # Rectangle-specific area calculation
        # This replaces the abstract method from Shape
        return self.width * self.height
    
    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.
        
        Overrides the parent class method with rectangle-specific calculation.
        
        Returns:
            float: Perimeter (2 * (width + height))
        """
        # Rectangle-specific perimeter calculation
        return 2 * (self.width + self.height)


class Circle(Shape):
    """
    A circle that inherits from Shape.
    
    Like Rectangle, Circle inherits from Shape but provides its own
    implementation of area() and perimeter(). This demonstrates how
    inheritance allows different classes to share a common interface
    while having different implementations.
    
    Attributes:
        radius (float): Radius of circle
    """
    
    def __init__(self, radius):
        """
        Initialize a circle.
        
        Args:
            radius (float): Radius
        """
        # Circle has a radius attribute (different from Rectangle's width/height)
        # Each subclass can have different attributes
        self.radius = radius
    
    def area(self):
        """
        Calculate the area of the circle.
        
        Overrides Shape.area() with circle-specific formula.
        Uses π * radius² for the calculation.
        
        Returns:
            float: Area of circle
        """
        # Import math module to access π (pi)
        import math
        
        # Circle area formula: π * r²
        # self.radius ** 2 squares the radius
        return math.pi * (self.radius ** 2)
    
    def perimeter(self):
        """
        Calculate the perimeter (circumference) of the circle.
        
        Overrides Shape.perimeter() with circle-specific formula.
        Uses 2 * π * radius for the calculation.
        
        Returns:
            float: Circumference of circle
        """
        # Import math module
        import math
        
        # Circle circumference formula: 2 * π * r
        return 2 * math.pi * self.radius


class OverdrawError(Exception):
    """
    Custom exception for account overdraw attempts.
    
    This demonstrates creating custom exceptions. In Python, exceptions
    are classes that inherit from the Exception base class (or one of its
    subclasses). Custom exceptions make error handling more specific and
    meaningful.
    
    By creating OverdrawError, we can catch this specific error type
    separately from other exceptions, allowing more precise error handling.
    """
    # The 'pass' statement means this class doesn't add anything
    # beyond what it inherits from Exception. It's a simple custom exception.
    # You could add custom attributes or methods here if needed.
    pass


class SafeBankAccount:
    """
    A bank account with exception handling.
    
    This demonstrates exception handling in methods. Instead of returning
    error codes or None, we raise exceptions when something goes wrong.
    This is Pythonic: "It's easier to ask for forgiveness than permission"
    (EAFP principle).
    
    Attributes:
        balance (float): Current account balance
    """
    
    def __init__(self, initial_balance=0.0):
        """
        Initialize a bank account.
        
        Args:
            initial_balance (float): Starting balance
        """
        # Store the initial balance
        self.balance = initial_balance
    
    def withdraw(self, amount):
        """
        Withdraw money from the account.
        
        This method demonstrates raising custom exceptions. If the withdrawal
        would cause an overdraw, we raise OverdrawError instead of silently
        failing or returning an error code.
        
        Args:
            amount (float): Amount to withdraw
        
        Raises:
            OverdrawError: If amount > balance
        
        Example:
            account = SafeBankAccount(100.0)
            account.withdraw(50.0)  # OK, balance becomes 50.0
            account.withdraw(100.0)  # Raises OverdrawError
        """
        # Check if withdrawal would cause overdraw
        if amount > self.balance:
            # Raise our custom exception with a descriptive message
            # This stops execution and propagates the error up the call stack
            # The caller can catch this exception and handle it appropriately
            raise OverdrawError("Insufficient funds")
        
        # If we get here, the withdrawal is safe
        # Subtract the amount from balance
        self.balance -= amount
    
    def get_balance(self):
        """
        Get the current balance.
        
        Returns:
            float: Current balance
        """
        return self.balance


def divide_numbers(a, b):
    """
    Divide two numbers with exception handling.
    
    This demonstrates the try/except pattern for handling exceptions.
    Instead of checking if b == 0 before dividing, we attempt the division
    and catch the ZeroDivisionError if it occurs. This is Python's EAFP
    (Easier to Ask for Forgiveness than Permission) philosophy.
    
    Args:
        a (float): Dividend
        b (float): Divisor
    
    Returns:
        float: Result of division, or None if division by zero
    
    Example:
        divide_numbers(10, 2)  # Returns 5.0
        divide_numbers(10, 0)   # Returns None (handles ZeroDivisionError)
    """
    # Try to perform the division
    # This is the "risky" code that might raise an exception
    try:
        result = a / b
        # If division succeeds, return the result
        return result
    
    # Catch the specific exception that might occur
    # ZeroDivisionError is raised when dividing by zero
    except ZeroDivisionError:
        # Handle the error: return None instead of crashing
        # In a real application, you might log the error or raise
        # a custom exception with more context
        return None
    
    # Note: We could also catch multiple exceptions:
    # except (ZeroDivisionError, TypeError) as e:
    #     return None
    
    # Or catch all exceptions (not recommended unless necessary):
    # except Exception as e:
    #     return None

