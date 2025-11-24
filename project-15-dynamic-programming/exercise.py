"""
Project 15: Dynamic Programming Fundamentals

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""


def fibonacci_memoization(n, memo=None):
    """
    Calculate nth Fibonacci number using memoization (top-down DP).
    
    Args:
        n (int): Fibonacci number to calculate
        memo (dict): Memoization cache (default None)
    
    Returns:
        int: nth Fibonacci number
    """
    # TODO: Initialize memo if None
    # TODO: Check if n in memo (base case)
    # TODO: If not, compute recursively and store in memo
    return None


def fibonacci_tabulation(n):
    """
    Calculate nth Fibonacci number using tabulation (bottom-up DP).
    
    Args:
        n (int): Fibonacci number to calculate
    
    Returns:
        int: nth Fibonacci number
    """
    # TODO: Handle base cases (n == 0 or n == 1)
    # TODO: Create table/array to store results
    # TODO: Fill table from bottom up
    # TODO: Return table[n]
    return None


def coin_change(coins, amount):
    """
    Find minimum number of coins needed to make amount.
    
    Uses dynamic programming to find optimal solution.
    
    Args:
        coins (list): Available coin denominations
        amount (int): Target amount
    
    Returns:
        int: Minimum coins needed, or -1 if impossible
    """
    # TODO: Create DP table
    # TODO: Initialize base case (0 coins for amount 0)
    # TODO: Fill table: for each amount, try each coin
    # TODO: Return result
    return None

