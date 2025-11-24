"""
Project 01: Basic Python Syntax and Variables

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""


def declare_variables():
    """
    Create and return variables of different types.
    
    Returns:
        tuple: A tuple containing (an_int, a_float, a_string, a_bool)
              where:
              - an_int is an integer with value 42
              - a_float is a float with value 3.14
              - a_string is a string with value "Python"
              - a_bool is a boolean with value True
    """
    # TODO: Create variables of each type
    an_int = None
    a_float = None
    a_string = None
    a_bool = None
    
    return an_int, a_float, a_string, a_bool


def perform_arithmetic(a, b):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        a (int or float): First number
        b (int or float): Second number
    
    Returns:
        tuple: A tuple containing (sum, difference, product, quotient)
              where:
              - sum is a + b
              - difference is a - b
              - product is a * b
              - quotient is a / b (float division)
    """
    # TODO: Perform arithmetic operations
    sum_result = None
    difference = None
    product = None
    quotient = None
    
    return sum_result, difference, product, quotient


def calculate_power_and_modulo(base, exponent, dividend, divisor):
    """
    Calculate power and modulo operations.
    
    Args:
        base (int): Base number
        exponent (int): Exponent
        dividend (int): Number to divide
        divisor (int): Number to divide by
    
    Returns:
        tuple: A tuple containing (power_result, modulo_result)
              where:
              - power_result is base raised to the power of exponent
              - modulo_result is the remainder when dividend is divided by divisor
    """
    # TODO: Calculate power and modulo
    power_result = None
    modulo_result = None
    
    return power_result, modulo_result


def convert_types(value_str, value_int, value_float):
    """
    Convert values between different types.
    
    Args:
        value_str (str): A string representation of a number (e.g., "42")
        value_int (int): An integer
        value_float (float): A float
    
    Returns:
        tuple: A tuple containing (str_to_int, int_to_str, float_to_int)
              where:
              - str_to_int is value_str converted to an integer
              - int_to_str is value_int converted to a string
              - float_to_int is value_float converted to an integer (truncated)
    """
    # TODO: Convert between types
    str_to_int = None
    int_to_str = None
    float_to_int = None
    
    return str_to_int, int_to_str, float_to_int


def string_operations(str1, str2, separator, count):
    """
    Perform string operations: concatenation and repetition.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        separator (str): String to repeat
        count (int): Number of times to repeat separator
    
    Returns:
        tuple: A tuple containing (concatenated, repeated)
              where:
              - concatenated is str1 and str2 joined with a space between them
              - repeated is separator repeated count times
    """
    # TODO: Perform string operations
    concatenated = None
    repeated = None
    
    return concatenated, repeated


def format_string(name, age, score):
    """
    Create a formatted string using f-string syntax.
    
    Args:
        name (str): A person's name
        age (int): Their age
        score (float): A score value
    
    Returns:
        str: A formatted string: "Hello, {name}! You are {age} years old and scored {score} points."
    """
    # TODO: Create formatted string using f-string
    formatted = None
    
    return formatted


def calculate_area_and_perimeter(length, width):
    """
    Calculate the area and perimeter of a rectangle.
    
    Args:
        length (float): Length of the rectangle
        width (float): Width of the rectangle
    
    Returns:
        tuple: A tuple containing (area, perimeter)
              where:
              - area is length * width
              - perimeter is 2 * (length + width)
    """
    # TODO: Calculate area and perimeter
    area = None
    perimeter = None
    
    return area, perimeter


def process_user_input(name_input, age_input, height_input):
    """
    Process user input strings and convert them to appropriate types.
    
    Args:
        name_input (str): User's name (already a string)
        age_input (str): User's age as a string (e.g., "25")
        height_input (str): User's height as a string (e.g., "5.9")
    
    Returns:
        tuple: A tuple containing (name, age, height)
              where:
              - name is the name_input as-is (string)
              - age is age_input converted to an integer
              - height is height_input converted to a float
    """
    # TODO: Process and convert inputs
    name = None
    age = None
    height = None
    
    return name, age, height

