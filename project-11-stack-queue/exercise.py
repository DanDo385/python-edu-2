"""
Project 11: Stack and Queue

Complete the class definitions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""


class Stack:
    """
    A stack implementation using a list.
    
    Stack is LIFO (Last In, First Out) - like a stack of plates.
    """
    
    def __init__(self):
        """Initialize an empty stack."""
        # TODO: Initialize self.items as empty list
        pass
    
    def push(self, item):
        """
        Add item to top of stack.
        
        Args:
            item: Item to add
        """
        # TODO: Add item to stack
        pass
    
    def pop(self):
        """
        Remove and return top item from stack.
        
        Returns:
            Item from top of stack
        
        Raises:
            IndexError: If stack is empty
        """
        # TODO: Remove and return top item
        return None
    
    def peek(self):
        """
        Return top item without removing it.
        
        Returns:
            Item from top of stack
        
        Raises:
            IndexError: If stack is empty
        """
        # TODO: Return top item without removing
        return None
    
    def is_empty(self):
        """
        Check if stack is empty.
        
        Returns:
            bool: True if empty, False otherwise
        """
        # TODO: Check if stack is empty
        return None
    
    def size(self):
        """
        Get number of items in stack.
        
        Returns:
            int: Number of items
        """
        # TODO: Return size of stack
        return None


class Queue:
    """
    A queue implementation using a list.
    
    Queue is FIFO (First In, First Out) - like a line of people.
    """
    
    def __init__(self):
        """Initialize an empty queue."""
        # TODO: Initialize self.items as empty list
        pass
    
    def enqueue(self, item):
        """
        Add item to back of queue.
        
        Args:
            item: Item to add
        """
        # TODO: Add item to back of queue
        pass
    
    def dequeue(self):
        """
        Remove and return front item from queue.
        
        Returns:
            Item from front of queue
        
        Raises:
            IndexError: If queue is empty
        """
        # TODO: Remove and return front item
        return None
    
    def peek(self):
        """
        Return front item without removing it.
        
        Returns:
            Item from front of queue
        
        Raises:
            IndexError: If queue is empty
        """
        # TODO: Return front item without removing
        return None
    
    def is_empty(self):
        """
        Check if queue is empty.
        
        Returns:
            bool: True if empty, False otherwise
        """
        # TODO: Check if queue is empty
        return None
    
    def size(self):
        """
        Get number of items in queue.
        
        Returns:
            int: Number of items
        """
        # TODO: Return size of queue
        return None


def is_balanced_parentheses(expression):
    """
    Check if parentheses are balanced using a stack.
    
    Balanced: ()(), ((())), (()())
    Not balanced: ((), ))(, ())
    
    Args:
        expression (str): String with parentheses
    
    Returns:
        bool: True if balanced, False otherwise
    """
    # TODO: Use stack to check balanced parentheses
    # Push opening brackets, pop when see closing bracket
    # Check if stack empty at end
    return None

