# Solution in Words: Dynamic Programming

## How to Think About This Problem

### Understanding Dynamic Programming

Think of DP like solving a puzzle - instead of solving the same piece multiple times, solve it once and remember the answer.

**Mental Model:**
```
Naive recursion: Solve same subproblem many times
DP: Solve once, remember answer, reuse it
```

### When to Use DP

1. **Overlapping subproblems**: Same calculations repeated
2. **Optimal substructure**: Best solution uses best sub-solutions
3. **Optimization problem**: Finding min/max/count

### Memoization (Top-Down)

**Thinking:**
- Start with recursive solution
- Add cache to store results
- Before computing, check cache
- After computing, store in cache

### Tabulation (Bottom-Up)

**Thinking:**
- Build solution from base cases up
- Use table/array to store results
- Fill table iteratively
- Return final table entry

### Problem-Solving Strategy

1. **Identify subproblems**: What are we solving repeatedly?
2. **Define state**: What information do we need?
3. **Find recurrence**: How do subproblems relate?
4. **Choose approach**: Memoization or tabulation?
5. **Implement**: Code the solution

