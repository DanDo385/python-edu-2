"""
Project 06: Python OOP Basics - SOLUTION

This file contains the complete solution with detailed comments explaining
each step of the implementation.
"""


class BankAccount:
    """
    A simple bank account class.
    
    This class demonstrates basic OOP concepts:
    - Constructor (__init__) to initialize objects
    - Instance attributes (self.balance)
    - Instance methods (deposit, withdraw, get_balance)
    - Encapsulation (data and methods together)
    
    Attributes:
        balance (float): Current account balance
    """
    
    def __init__(self, initial_balance=0.0):
        """
        Initialize a bank account with an initial balance.
        
        The __init__ method is called automatically when you create a new
        BankAccount object. It's like a constructor in other languages.
        
        Args:
            initial_balance (float): Starting balance (default 0.0)
        
        Example:
            account = BankAccount(100.0)  # Creates account with $100
            account = BankAccount()       # Creates account with $0 (default)
        """
        # Set the instance attribute self.balance
        # 'self' refers to the specific instance of BankAccount being created
        # Each BankAccount object has its own balance attribute
        self.balance = initial_balance
    
    def deposit(self, amount):
        """
        Deposit money into the account.
        
        This method modifies the account's balance by adding the deposit amount.
        Methods that modify object state are called "mutator methods" or "setters".
        
        Args:
            amount (float): Amount to deposit
        
        Example:
            account = BankAccount(100.0)
            account.deposit(50.0)  # Balance becomes 150.0
        """
        # Add the deposit amount to the current balance
        # self.balance accesses the instance's balance attribute
        # += is shorthand for: self.balance = self.balance + amount
        self.balance += amount
    
    def withdraw(self, amount):
        """
        Withdraw money from the account.
        
        This method modifies the account's balance by subtracting the withdrawal amount.
        Note: In a real application, you'd want to check if there's enough balance
        before allowing withdrawal. For this exercise, we keep it simple.
        
        Args:
            amount (float): Amount to withdraw
        
        Example:
            account = BankAccount(100.0)
            account.withdraw(30.0)  # Balance becomes 70.0
        """
        # Subtract the withdrawal amount from the current balance
        # -= is shorthand for: self.balance = self.balance - amount
        self.balance -= amount
    
    def get_balance(self):
        """
        Get the current account balance.
        
        This is an "accessor method" or "getter" - it returns information
        about the object without modifying it. Methods that only return
        values are called "pure" methods.
        
        Returns:
            float: Current balance
        
        Example:
            account = BankAccount(100.0)
            balance = account.get_balance()  # Returns 100.0
        """
        # Return the current balance
        # This allows external code to read the balance without direct access
        # (though Python doesn't enforce private attributes like some languages)
        return self.balance


class Point:
    """
    A point in 2D space.
    
    This class demonstrates:
    - Multiple attributes (x, y coordinates)
    - Calculation methods (distance calculations)
    - Using external modules (math) within methods
    
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
        
        Example:
            point = Point(3.0, 4.0)  # Point at (3, 4)
        """
        # Store the x and y coordinates as instance attributes
        # Each Point object has its own x and y values
        self.x = x
        self.y = y
    
    def distance_from_origin(self):
        """
        Calculate distance from origin (0, 0).
        
        Uses the Pythagorean theorem: distance = sqrt(x² + y²)
        The origin is at coordinates (0, 0).
        
        Returns:
            float: Distance from origin
        
        Example:
            point = Point(3.0, 4.0)
            dist = point.distance_from_origin()  # Returns 5.0
        """
        # Import math module to use sqrt function
        # We import here (could also import at top of file)
        import math
        
        # Calculate distance using Pythagorean theorem
        # x**2 means x squared (x to the power of 2)
        # math.sqrt calculates the square root
        distance = math.sqrt(self.x**2 + self.y**2)
        
        return distance
    
    def distance_to(self, other_point):
        """
        Calculate distance to another point.
        
        Uses the distance formula: sqrt((x2-x1)² + (y2-y1)²)
        This is the standard Euclidean distance between two points.
        
        Args:
            other_point (Point): Another point
        
        Returns:
            float: Distance between points
        
        Example:
            point1 = Point(0.0, 0.0)
            point2 = Point(3.0, 4.0)
            dist = point1.distance_to(point2)  # Returns 5.0
        """
        # Import math module
        import math
        
        # Calculate differences in x and y coordinates
        # self.x and self.y are this point's coordinates
        # other_point.x and other_point.y are the other point's coordinates
        dx = self.x - other_point.x  # Difference in x
        dy = self.y - other_point.y  # Difference in y
        
        # Calculate distance using distance formula
        # (dx)**2 + (dy)**2 gives the squared distance
        # math.sqrt gives us the actual distance
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance


class Rectangle:
    """
    A rectangle with width and height.
    
    This class demonstrates:
    - Geometric calculations
    - Multiple calculation methods
    - Simple attribute storage
    
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
        
        Example:
            rect = Rectangle(5.0, 3.0)  # 5 units wide, 3 units tall
        """
        # Store width and height as instance attributes
        self.width = width
        self.height = height
    
    def area(self):
        """
        Calculate the area of the rectangle.
        
        Formula: area = width × height
        
        Returns:
            float: Area (width * height)
        
        Example:
            rect = Rectangle(5.0, 3.0)
            area = rect.area()  # Returns 15.0
        """
        # Area is simply width multiplied by height
        # This is a calculation based on the object's attributes
        return self.width * self.height
    
    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.
        
        Formula: perimeter = 2 × (width + height)
        Perimeter is the distance around the rectangle.
        
        Returns:
            float: Perimeter (2 * (width + height))
        
        Example:
            rect = Rectangle(5.0, 3.0)
            perimeter = rect.perimeter()  # Returns 16.0
        """
        # Perimeter is 2 times the sum of width and height
        # Parentheses ensure addition happens before multiplication
        return 2 * (self.width + self.height)


class Student:
    """
    A student with name and grades.
    
    This class demonstrates:
    - Managing collections (list of grades)
    - Adding items to collections
    - Calculating statistics (average)
    
    Attributes:
        name (str): Student's name
        grades (list): List of grade scores
    """
    
    def __init__(self, name):
        """
        Initialize a student.
        
        Args:
            name (str): Student's name
        
        Example:
            student = Student("Alice")
        """
        # Store the student's name
        self.name = name
        
        # Initialize grades as an empty list
        # We'll add grades later using the add_grade method
        # Each Student object has its own independent list of grades
        self.grades = []
    
    def add_grade(self, grade):
        """
        Add a grade to the student's record.
        
        This method modifies the student's grades list by appending
        a new grade. The list grows as more grades are added.
        
        Args:
            grade (float): Grade score to add
        
        Example:
            student = Student("Alice")
            student.add_grade(85.0)
            student.add_grade(90.0)
        """
        # Append the new grade to the grades list
        # list.append() adds an item to the end of the list
        # This modifies the list in-place (mutates it)
        self.grades.append(grade)
    
    def get_average(self):
        """
        Calculate the average of all grades.
        
        Formula: average = sum of all grades / number of grades
        If there are no grades, return 0.0 to avoid division by zero.
        
        Returns:
            float: Average grade, or 0.0 if no grades
        
        Example:
            student = Student("Alice")
            student.add_grade(80.0)
            student.add_grade(90.0)
            student.add_grade(100.0)
            avg = student.get_average()  # Returns 90.0
        """
        # Check if there are any grades
        # If the list is empty, we can't calculate an average
        if len(self.grades) == 0:
            return 0.0
        
        # Calculate the sum of all grades
        # sum() is a built-in function that adds all items in a list
        total = sum(self.grades)
        
        # Calculate average: total divided by number of grades
        # len() returns the number of items in the list
        average = total / len(self.grades)
        
        return average

