"""
Project 01: Basic Python Syntax and Variables - SOLUTION

This file contains the complete solution with detailed comments explaining
each step of the implementation.
"""


def declare_variables():
    """
    Create and return variables of different types.
    
    In Python, variables are created simply by assignment. No declaration
    is needed - Python infers the type from the value assigned.
    
    Returns:
        tuple: A tuple containing (an_int, a_float, a_string, a_bool)
    """
    # Integer: Whole numbers, positive or negative
    # Python integers can be arbitrarily large (unlike some languages)
    an_int = 42
    
    # Float: Decimal numbers
    # Note: Python uses double-precision floating point (64-bit)
    # Floats are approximations - 0.1 + 0.2 might not equal 0.3 exactly
    a_float = 3.14
    
    # String: Sequence of characters
    # Can use single quotes, double quotes, or triple quotes
    # Triple quotes allow multi-line strings
    a_string = "Python"
    
    # Boolean: True or False (capitalized in Python)
    # Note: In Python, these are actually subclasses of int (True == 1, False == 0)
    # but they're used for logical operations
    a_bool = True
    
    return an_int, a_float, a_string, a_bool


def perform_arithmetic(a, b):
    """
    Perform basic arithmetic operations on two numbers.
    
    Python's arithmetic operators work as expected, but division behavior
    is important: / always returns a float in Python 3, even for integers.
    
    Args:
        a (int or float): First number
        b (int or float): Second number
    
    Returns:
        tuple: A tuple containing (sum, difference, product, quotient)
    """
    # Addition: Simply add the two numbers
    # Works with both int and float types
    sum_result = a + b
    
    # Subtraction: Subtract second number from first
    difference = a - b
    
    # Multiplication: Multiply the two numbers
    # Note: Can also use * for string repetition (e.g., "a" * 3 = "aaa")
    product = a * b
    
    # Division: Divide first by second
    # CRITICAL: In Python 3, / always returns a float
    # Even 10 / 2 returns 5.0, not 5
    # For integer division (floor division), use // instead
    quotient = a / b
    
    return sum_result, difference, product, quotient


def calculate_power_and_modulo(base, exponent, dividend, divisor):
    """
    Calculate power and modulo operations.
    
    Power: base raised to exponent (base^exponent)
    Modulo: Remainder after division (dividend % divisor)
    
    Args:
        base (int): Base number
        exponent (int): Exponent
        dividend (int): Number to divide
        divisor (int): Number to divide by
    
    Returns:
        tuple: A tuple containing (power_result, modulo_result)
    """
    # Exponentiation: Use ** operator
    # This is equivalent to base^exponent in mathematics
    # Example: 2 ** 8 = 256 (2 to the power of 8)
    # Note: Can also use pow(base, exponent) function
    power_result = base ** exponent
    
    # Modulo: Returns the remainder after division
    # Example: 10 % 3 = 1 (because 10 / 3 = 3 remainder 1)
    # Very useful for:
    # - Checking if a number is even/odd (n % 2 == 0 means even)
    # - Wrapping around cycles (e.g., array indices)
    # - Finding patterns in sequences
    modulo_result = dividend % divisor
    
    return power_result, modulo_result


def convert_types(value_str, value_int, value_float):
    """
    Convert values between different types.
    
    Type conversion is crucial when working with user input (which is
    always a string) or when you need to format numbers as text.
    
    Args:
        value_str (str): A string representation of a number (e.g., "42")
        value_int (int): An integer
        value_float (float): A float
    
    Returns:
        tuple: A tuple containing (str_to_int, int_to_str, float_to_int)
    """
    # String to Integer: Use int() constructor
    # This parses the string and converts it to an integer
    # Will raise ValueError if string doesn't represent a valid integer
    # Example: int("42") → 42, int("-10") → -10
    # Note: int("3.14") would raise ValueError - need to convert to float first
    str_to_int = int(value_str)
    
    # Integer to String: Use str() constructor
    # Converts any value to its string representation
    # Essential for concatenating numbers with text
    # Example: str(42) → "42"
    int_to_str = str(value_int)
    
    # Float to Integer: Use int() on a float
    # IMPORTANT: This truncates (cuts off decimal part), doesn't round
    # Example: int(3.14) → 3, int(3.99) → 3, int(-3.7) → -3
    # To round instead: use round(3.7) → 4, or int(round(3.7))
    float_to_int = int(value_float)
    
    return str_to_int, int_to_str, float_to_int


def string_operations(str1, str2, separator, count):
    """
    Perform string operations: concatenation and repetition.
    
    Strings in Python are immutable (can't be changed in place), but
    you can create new strings through operations.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        separator (str): String to repeat
        count (int): Number of times to repeat separator
    
    Returns:
        tuple: A tuple containing (concatenated, repeated)
    """
    # String Concatenation: Use + operator
    # Joins two strings together
    # Note: Can only concatenate strings with strings
    # To concatenate with numbers, convert numbers to strings first
    # Example: "Hello" + " " + "World" → "Hello World"
    concatenated = str1 + " " + str2
    
    # String Repetition: Use * operator
    # Repeats a string a specified number of times
    # Very useful for creating separators, borders, etc.
    # Example: "-" * 5 → "-----"
    # Note: count must be an integer (or something that can be converted)
    repeated = separator * count
    
    return concatenated, repeated


def format_string(name, age, score):
    """
    Create a formatted string using f-string syntax.
    
    F-strings (formatted string literals) are Python's modern way to
    format strings. They're introduced in Python 3.6+ and are preferred
    over older methods like .format() or % formatting.
    
    Syntax: f"text {variable} more text"
    The expressions inside {} are evaluated and inserted into the string.
    
    Args:
        name (str): A person's name
        age (int): Their age
        score (float): A score value
    
    Returns:
        str: A formatted string
    """
    # F-string: Prefix the string with 'f' or 'F'
    # Inside curly braces {}, you can put any Python expression
    # Python will evaluate it and convert to string automatically
    # Much more readable than: "Hello, " + name + "! You are " + str(age) + "..."
    
    # Format: f"text {expression} text {expression} ..."
    # The expressions are evaluated at runtime
    # Numbers are automatically converted to strings
    # You can even do calculations: f"Next year: {age + 1}"
    formatted = f"Hello, {name}! You are {age} years old and scored {score} points."
    
    # Alternative methods (for reference, not used here):
    # 1. .format() method: "Hello, {}!".format(name)
    # 2. % formatting: "Hello, %s!" % name (old style, avoid)
    # 3. Concatenation: "Hello, " + name + "!" (verbose, avoid)
    
    return formatted


def calculate_area_and_perimeter(length, width):
    """
    Calculate the area and perimeter of a rectangle.
    
    This demonstrates using variables in calculations and understanding
    the formulas for geometric shapes.
    
    Args:
        length (float): Length of the rectangle
        width (float): Width of the rectangle
    
    Returns:
        tuple: A tuple containing (area, perimeter)
    """
    # Area Formula: length × width
    # This gives the space inside the rectangle (in square units)
    # Example: length=5, width=3 → area = 15
    area = length * width
    
    # Perimeter Formula: 2 × (length + width)
    # This gives the distance around the rectangle (in linear units)
    # Example: length=5, width=3 → perimeter = 2 × (5 + 3) = 16
    # Note: Parentheses ensure addition happens before multiplication
    perimeter = 2 * (length + width)
    
    return area, perimeter


def process_user_input(name_input, age_input, height_input):
    """
    Process user input strings and convert them to appropriate types.
    
    IMPORTANT: The input() function always returns a string, even if
    the user types a number. You must convert it to the desired type.
    
    This function simulates what happens when you read user input:
    - name_input: Already a string (no conversion needed)
    - age_input: String like "25" needs conversion to int
    - height_input: String like "5.9" needs conversion to float
    
    Args:
        name_input (str): User's name (already a string)
        age_input (str): User's age as a string (e.g., "25")
        height_input (str): User's height as a string (e.g., "5.9")
    
    Returns:
        tuple: A tuple containing (name, age, height)
    """
    # Name: Already a string, no conversion needed
    # When using input("Enter name: "), it returns a string directly
    name = name_input
    
    # Age: Convert string to integer
    # User types "25", but input() gives us the string "25"
    # We need int() to convert it to the number 25
    # This allows us to do math with it later (e.g., age + 1)
    # Will raise ValueError if string can't be converted (e.g., "abc")
    age = int(age_input)
    
    # Height: Convert string to float
    # User types "5.9", but input() gives us the string "5.9"
    # We need float() to convert it to the decimal number 5.9
    # Use float() for any decimal input (prices, measurements, etc.)
    # Will raise ValueError if string can't be converted
    height = float(height_input)
    
    # Real-world usage example:
    # name = input("Enter your name: ")
    # age = int(input("Enter your age: "))
    # height = float(input("Enter your height in feet: "))
    
    return name, age, height

