"""
Project 04: Data Structures I – Lists and Tuples - SOLUTION

Complete solution with detailed comments.
"""

def find_minimum(numbers):
    """Find minimum using loop comparison."""
    if not numbers:
        return None
    min_val = numbers[0]
    for num in numbers[1:]:
        if num < min_val:
            min_val = num
    return min_val

def find_maximum(numbers):
    """Find maximum using loop comparison."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val

def calculate_average(numbers):
    """Calculate average: sum divided by count."""
    if not numbers:
        return None
    total = sum(numbers)
    return total / len(numbers)

def merge_sorted_lists(list1, list2):
    """Merge using two pointers approach."""
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

def reverse_list(items):
    """Reverse in-place by swapping from ends."""
    left, right = 0, len(items) - 1
    while left < right:
        items[left], items[right] = items[right], items[left]
        left += 1
        right -= 1

def remove_duplicates(items):
    """Remove duplicates preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def list_comprehension_squares(n):
    """Generate squares using list comprehension."""
    return [x**2 for x in range(n)]

def get_even_numbers(numbers):
    """Filter evens using list comprehension."""
    return [x for x in numbers if x % 2 == 0]

def tuple_operations():
    """Demonstrate tuple creation and unpacking."""
    my_tuple = (3, 4)
    a, b = my_tuple
    return a + b, a * b

