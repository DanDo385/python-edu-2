"""
Test suite for Project 01: Basic Python Syntax and Variables

Run with: pytest test.py -v
"""

import pytest
from exercise import (
    declare_variables,
    perform_arithmetic,
    calculate_power_and_modulo,
    convert_types,
    string_operations,
    format_string,
    calculate_area_and_perimeter,
    process_user_input,
)


class TestDeclareVariables:
    """Test variable declaration and types."""
    
    def test_variable_types(self):
        """Test that variables are declared with correct types."""
        an_int, a_float, a_string, a_bool = declare_variables()
        
        assert isinstance(an_int, int), "an_int should be an integer"
        assert isinstance(a_float, float), "a_float should be a float"
        assert isinstance(a_string, str), "a_string should be a string"
        assert isinstance(a_bool, bool), "a_bool should be a boolean"
    
    def test_variable_values(self):
        """Test that variables have correct values."""
        an_int, a_float, a_string, a_bool = declare_variables()
        
        assert an_int == 42, "an_int should be 42"
        assert a_float == 3.14, "a_float should be 3.14"
        assert a_string == "Python", "a_string should be 'Python'"
        assert a_bool is True, "a_bool should be True"


class TestArithmetic:
    """Test arithmetic operations."""
    
    def test_addition(self):
        """Test addition operation."""
        sum_result, _, _, _ = perform_arithmetic(10, 5)
        assert sum_result == 15, "10 + 5 should equal 15"
    
    def test_subtraction(self):
        """Test subtraction operation."""
        _, difference, _, _ = perform_arithmetic(10, 5)
        assert difference == 5, "10 - 5 should equal 5"
    
    def test_multiplication(self):
        """Test multiplication operation."""
        _, _, product, _ = perform_arithmetic(10, 5)
        assert product == 50, "10 * 5 should equal 50"
    
    def test_division(self):
        """Test division operation (float division)."""
        _, _, _, quotient = perform_arithmetic(10, 3)
        assert abs(quotient - 3.3333333333333335) < 0.0001, "10 / 3 should be approximately 3.333..."
        assert isinstance(quotient, float), "Division should return a float"
    
    def test_negative_numbers(self):
        """Test arithmetic with negative numbers."""
        sum_result, difference, product, quotient = perform_arithmetic(-5, 3)
        assert sum_result == -2, "-5 + 3 should equal -2"
        assert difference == -8, "-5 - 3 should equal -8"
        assert product == -15, "-5 * 3 should equal -15"
        assert abs(quotient - (-1.6666666666666667)) < 0.0001, "-5 / 3 should be approximately -1.666..."


class TestPowerAndModulo:
    """Test power and modulo operations."""
    
    def test_power(self):
        """Test exponentiation."""
        power_result, _ = calculate_power_and_modulo(2, 8, 0, 1)
        assert power_result == 256, "2 ** 8 should equal 256"
    
    def test_modulo(self):
        """Test modulo operation."""
        _, modulo_result = calculate_power_and_modulo(0, 0, 10, 3)
        assert modulo_result == 1, "10 % 3 should equal 1"
    
    def test_power_zero(self):
        """Test power with zero exponent."""
        power_result, _ = calculate_power_and_modulo(5, 0, 0, 1)
        assert power_result == 1, "Any number to the power of 0 should be 1"
    
    def test_modulo_edge_cases(self):
        """Test modulo with edge cases."""
        _, modulo_result = calculate_power_and_modulo(0, 0, 15, 5)
        assert modulo_result == 0, "15 % 5 should equal 0"


class TestTypeConversion:
    """Test type conversions."""
    
    def test_string_to_int(self):
        """Test converting string to integer."""
        str_to_int, _, _ = convert_types("42", 0, 0.0)
        assert str_to_int == 42, "'42' should convert to integer 42"
        assert isinstance(str_to_int, int), "Result should be an integer"
    
    def test_int_to_string(self):
        """Test converting integer to string."""
        _, int_to_str, _ = convert_types("0", 42, 0.0)
        assert int_to_str == "42", "42 should convert to string '42'"
        assert isinstance(int_to_str, str), "Result should be a string"
    
    def test_float_to_int(self):
        """Test converting float to integer (truncation)."""
        _, _, float_to_int = convert_types("0", 0, 3.14)
        assert float_to_int == 3, "3.14 should truncate to integer 3"
        assert isinstance(float_to_int, int), "Result should be an integer"
    
    def test_negative_conversions(self):
        """Test conversions with negative numbers."""
        str_to_int, int_to_str, float_to_int = convert_types("-10", -5, -3.7)
        assert str_to_int == -10, "Should handle negative strings"
        assert int_to_str == "-5", "Should handle negative integers"
        assert float_to_int == -3, "Should truncate negative floats correctly"


class TestStringOperations:
    """Test string operations."""
    
    def test_concatenation(self):
        """Test string concatenation."""
        concatenated, _ = string_operations("Hello", "World", "-", 0)
        assert concatenated == "Hello World", "Should concatenate with space"
    
    def test_repetition(self):
        """Test string repetition."""
        _, repeated = string_operations("", "", "-", 5)
        assert repeated == "-----", "Should repeat '-' 5 times"
    
    def test_both_operations(self):
        """Test both string operations together."""
        concatenated, repeated = string_operations("Python", "Rocks", "*", 3)
        assert concatenated == "Python Rocks", "Should concatenate correctly"
        assert repeated == "***", "Should repeat '*' 3 times"


class TestFormatString:
    """Test string formatting."""
    
    def test_f_string_formatting(self):
        """Test f-string formatting."""
        result = format_string("Alice", 25, 95.5)
        expected = "Hello, Alice! You are 25 years old and scored 95.5 points."
        assert result == expected, "F-string should format correctly"
    
    def test_different_values(self):
        """Test formatting with different values."""
        result = format_string("Bob", 30, 87.25)
        expected = "Hello, Bob! You are 30 years old and scored 87.25 points."
        assert result == expected, "Should format with different values correctly"


class TestAreaAndPerimeter:
    """Test area and perimeter calculations."""
    
    def test_rectangle_calculations(self):
        """Test rectangle area and perimeter."""
        area, perimeter = calculate_area_and_perimeter(5.0, 3.0)
        assert area == 15.0, "Area of 5x3 rectangle should be 15"
        assert perimeter == 16.0, "Perimeter of 5x3 rectangle should be 16"
    
    def test_square(self):
        """Test square (equal length and width)."""
        area, perimeter = calculate_area_and_perimeter(4.0, 4.0)
        assert area == 16.0, "Area of 4x4 square should be 16"
        assert perimeter == 16.0, "Perimeter of 4x4 square should be 16"
    
    def test_decimal_values(self):
        """Test with decimal values."""
        area, perimeter = calculate_area_and_perimeter(2.5, 1.5)
        assert abs(area - 3.75) < 0.001, "Area should be 3.75"
        assert abs(perimeter - 8.0) < 0.001, "Perimeter should be 8.0"


class TestProcessUserInput:
    """Test processing user input."""
    
    def test_input_processing(self):
        """Test converting input strings to appropriate types."""
        name, age, height = process_user_input("Alice", "25", "5.9")
        assert name == "Alice", "Name should remain a string"
        assert age == 25, "Age should be converted to integer"
        assert isinstance(age, int), "Age should be an integer type"
        assert height == 5.9, "Height should be converted to float"
        assert isinstance(height, float), "Height should be a float type"
    
    def test_different_inputs(self):
        """Test with different input values."""
        name, age, height = process_user_input("Bob", "30", "6.2")
        assert name == "Bob"
        assert age == 30
        assert height == 6.2

