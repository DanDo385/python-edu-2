# Project 15: Dynamic Programming Fundamentals

## Learning Objectives

- Understand overlapping subproblems and optimal substructure
- Implement memoization (top-down DP)
- Implement tabulation (bottom-up DP)
- Solve classic DP problems (Fibonacci, coin change)
- Recognize when to use dynamic programming

## Problem Description

Dynamic Programming optimizes recursive solutions by storing results of subproblems. It's essential for solving optimization problems efficiently.

## Key Concepts

### DP Characteristics

- **Overlapping subproblems**: Same subproblems computed multiple times
- **Optimal substructure**: Optimal solution contains optimal solutions to subproblems

### Approaches

- **Memoization**: Top-down, cache results as you compute
- **Tabulation**: Bottom-up, build table from base cases

## How Python Uniquely Solves This

Python dictionaries make memoization natural. List comprehensions help with tabulation. Decorators can automate memoization.

## Comparison with Other Languages

- **Go**: Similar approach with maps for memoization
- **TypeScript**: Similar, with optional type annotations
- **Rust**: Ownership makes DP implementations more complex

## Detailed Problem Analysis

### When to Use Dynamic Programming

**Two Key Characteristics:**
1. **Overlapping Subproblems**: Same subproblems solved multiple times
2. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems

**Example - Fibonacci:**
```
Naive recursion:
fib(5) = fib(4) + fib(3)
       = (fib(3) + fib(2)) + (fib(2) + fib(1))
       = ((fib(2) + fib(1)) + fib(2)) + (fib(2) + fib(1))
       = ... (fib(2) computed 3 times!)
```

### Memoization (Top-Down DP)

**Approach:** Cache results as you compute them recursively

**Fibonacci with Memoization:**
```python
memo = {}
def fib(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
```

**Time Complexity:** O(n) - each subproblem computed once
**Space Complexity:** O(n) - memo dictionary + recursion stack

**Advantages:**
- Natural recursive structure
- Only computes needed subproblems
- Easy to understand

**Disadvantages:**
- Recursion overhead
- Stack overflow risk for deep recursion

### Tabulation (Bottom-Up DP)

**Approach:** Build table from base cases up

**Fibonacci with Tabulation:**
```python
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**Time Complexity:** O(n) - single pass through array
**Space Complexity:** O(n) - DP table (can optimize to O(1))

**Advantages:**
- No recursion overhead
- Can optimize space (only need last 2 values)
- Guaranteed to compute all subproblems

**Disadvantages:**
- May compute unnecessary subproblems
- Less intuitive than recursion

### Coin Change Problem

**Problem:** Find minimum coins needed to make amount

**Example:** coins = [1, 3, 4], amount = 6
- Greedy (wrong): 4 + 1 + 1 = 3 coins
- DP (correct): 3 + 3 = 2 coins

**DP Approach:**
```
dp[i] = minimum coins to make amount i

Base case: dp[0] = 0 (0 coins for 0 amount)
For each amount from 1 to target:
  For each coin:
    if coin <= amount:
      dp[amount] = min(dp[amount], 1 + dp[amount - coin])
```

**Time Complexity:** O(amount × coins)
**Space Complexity:** O(amount)

### Memoization vs Tabulation

| Aspect | Memoization | Tabulation |
|--------|-------------|------------|
| Direction | Top-down | Bottom-up |
| Approach | Recursive | Iterative |
| Space | O(n) + stack | O(n) |
| Computes | Only needed | All subproblems |
| Intuition | More natural | Less intuitive |

### Edge Cases to Handle

1. **Base cases**: Handle n=0, n=1, amount=0
2. **Impossible problems**: No solution exists (return -1 or infinity)
3. **Negative values**: Validate inputs
4. **Empty arrays**: Handle empty coin list
5. **Large values**: Consider integer overflow

### Common Mistakes

1. **Not initializing base cases**: Incorrect starting values
2. **Off-by-one errors**: Array indexing mistakes
3. **Not handling impossible cases**: Forgetting to check if solution exists
4. **Wrong recurrence relation**: Incorrect DP formula
5. **Not optimizing space**: Using O(n) when O(1) possible

## Deliverables

Complete DP implementations with comprehensive solutions:

1. **Fibonacci - Memoization** - Top-down approach with cache
2. **Fibonacci - Tabulation** - Bottom-up approach with array
3. **Fibonacci - Space Optimized** - O(1) space using variables
4. **Coin Change - Minimum Coins** - Find minimum coins needed
5. **Coin Change - Count Ways** - Count number of ways to make amount
6. **Performance Comparison** - Compare naive recursion vs memoization vs tabulation
7. **Longest Increasing Subsequence** - Classic DP problem
8. **Climbing Stairs** - Simple DP problem (similar to Fibonacci)

**Additional Challenges:**
- Edit distance (Levenshtein distance)
- Longest common subsequence (LCS)
- 0/1 Knapsack problem
- House robber problem
- Unique paths in grid

