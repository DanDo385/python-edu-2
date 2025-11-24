"""
Project 15: Dynamic Programming Fundamentals - SOLUTION

Complete solution with detailed comments explaining memoization, tabulation, and DP principles.
"""


def fibonacci_memoization(n, memo=None):
    """
    Calculate nth Fibonacci number using memoization (top-down DP).
    
    Memoization is a top-down approach: we start with the problem we want
    to solve and work down to base cases, caching results as we go.
    
    Key insight: fibonacci(n) calls fibonacci(n-1) and fibonacci(n-2),
    which in turn call fibonacci(n-2), fibonacci(n-3), etc. Many of these
    calls are repeated! Memoization stores results to avoid recomputation.
    
    Time complexity: O(n) - each fibonacci(i) computed once
    Space complexity: O(n) - memo dictionary and recursion stack
    
    Args:
        n (int): Fibonacci number to calculate
        memo (dict): Memoization cache (default None, creates new dict)
    
    Returns:
        int: nth Fibonacci number
    
    Example:
        fibonacci_memoization(10)  # Returns 55
    """
    # Initialize memo dictionary if not provided
    # Using default parameter allows recursive calls to share same cache
    if memo is None:
        memo = {}
    
    # Base cases: F(0) = 0, F(1) = 1
    # These don't need recursion
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Check if we've already computed this value
    # This is the key to memoization: avoid recomputation
    if n in memo:
        return memo[n]
    
    # Compute fibonacci(n) recursively
    # F(n) = F(n-1) + F(n-2)
    # Store result in memo before returning
    memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    
    return memo[n]


def fibonacci_tabulation(n):
    """
    Calculate nth Fibonacci number using tabulation (bottom-up DP).
    
    Tabulation is a bottom-up approach: we start with base cases and build
    up to the solution we want. We use a table (array) to store results.
    
    Key insight: Instead of recursion, we iterate from base cases upward,
    filling a table with results. This avoids recursion overhead and
    can be more space-efficient.
    
    Time complexity: O(n) - single pass through table
    Space complexity: O(n) - table size, but can be optimized to O(1)
    
    Args:
        n (int): Fibonacci number to calculate
    
    Returns:
        int: nth Fibonacci number
    
    Example:
        fibonacci_tabulation(10)  # Returns 55
    """
    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Create table to store Fibonacci numbers
    # dp[i] will store the ith Fibonacci number
    # We need n+1 elements (0 to n)
    dp = [0] * (n + 1)
    
    # Initialize base cases
    dp[0] = 0
    dp[1] = 1
    
    # Fill table from bottom up
    # For each i from 2 to n, compute dp[i] using previous values
    for i in range(2, n + 1):
        # Fibonacci recurrence: F(i) = F(i-1) + F(i-2)
        # We've already computed dp[i-1] and dp[i-2]
        dp[i] = dp[i - 1] + dp[i - 2]
    
    # Return the nth Fibonacci number
    return dp[n]
    
    # Note: We could optimize space to O(1) by only keeping last two values:
    # prev2, prev1 = 0, 1
    # for i in range(2, n+1):
    #     curr = prev1 + prev2
    #     prev2, prev1 = prev1, curr
    # return prev1


def coin_change(coins, amount):
    """
    Find minimum number of coins needed to make amount.
    
    This is a classic DP problem. We want to find the minimum number of
    coins to make a given amount using given coin denominations.
    
    DP approach:
    - Subproblem: minimum coins for amount i
    - State: dp[i] = minimum coins needed for amount i
    - Recurrence: dp[i] = min(dp[i], dp[i - coin] + 1) for each coin
    - Base case: dp[0] = 0 (0 coins for amount 0)
    
    Time complexity: O(amount * len(coins))
    Space complexity: O(amount) for DP table
    
    Args:
        coins (list): Available coin denominations (e.g., [1, 3, 4])
        amount (int): Target amount
    
    Returns:
        int: Minimum coins needed, or -1 if impossible
    
    Example:
        coin_change([1, 3, 4], 6)  # Returns 2 (3 + 3)
        coin_change([2], 3)         # Returns -1 (impossible)
    """
    # Create DP table
    # dp[i] = minimum coins needed to make amount i
    # Initialize with a large number (infinity) to represent "impossible"
    # We use amount + 1 as a sentinel (more than any possible solution)
    dp = [amount + 1] * (amount + 1)
    
    # Base case: 0 coins needed to make amount 0
    dp[0] = 0
    
    # Fill table from 1 to amount
    # For each amount, try each coin denomination
    for i in range(1, amount + 1):
        # Try each coin
        for coin in coins:
            # Check if we can use this coin (coin value <= current amount)
            if coin <= i:
                # Update dp[i] if using this coin gives better solution
                # dp[i - coin] is minimum coins for remaining amount
                # +1 because we're using one coin
                # min() keeps the best solution so far
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    # If dp[amount] is still the sentinel value, it's impossible
    # Otherwise, return the minimum coins
    if dp[amount] > amount:
        return -1
    
    return dp[amount]

