"""
Project 11: Stack and Queue - SOLUTION

Complete solution with detailed comments explaining LIFO and FIFO principles.
"""


class Stack:
    """
    A stack implementation using a list.
    
    Stack is LIFO (Last In, First Out) - the last item added is the first
    item removed. Think of it like a stack of plates: you add plates to
    the top and remove from the top.
    
    We use a list where the "top" is the end of the list. This makes
    push/pop operations O(1) amortized.
    """
    
    def __init__(self):
        """
        Initialize an empty stack.
        
        We use a list to store stack items. The end of the list represents
        the top of the stack.
        """
        # Initialize empty list to store stack items
        # We'll use append() to push (add to end) and pop() to pop (remove from end)
        self.items = []
    
    def push(self, item):
        """
        Add item to top of stack.
        
        Since we're using a list and the "top" is the end, we append
        to the end of the list. This is O(1) amortized.
        
        Args:
            item: Item to add
        """
        # Add item to end of list (top of stack)
        # append() adds to the end, which represents the top
        self.items.append(item)
    
    def pop(self):
        """
        Remove and return top item from stack.
        
        Removes from the end of the list (top of stack) and returns it.
        This is O(1) since we're removing from the end.
        
        Returns:
            Item from top of stack
        
        Raises:
            IndexError: If stack is empty
        """
        # Remove and return last item (top of stack)
        # pop() without arguments removes and returns the last element
        # This raises IndexError if list is empty, which is appropriate
        return self.items.pop()
    
    def peek(self):
        """
        Return top item without removing it.
        
        Just looks at the last element without modifying the stack.
        This is O(1).
        
        Returns:
            Item from top of stack
        
        Raises:
            IndexError: If stack is empty
        """
        # Return last item without removing it
        # [-1] accesses the last element (top of stack)
        # This raises IndexError if list is empty
        return self.items[-1]
    
    def is_empty(self):
        """
        Check if stack is empty.
        
        Returns:
            bool: True if empty, False otherwise
        """
        # Check if list is empty
        # len(self.items) == 0 would also work, but this is more Pythonic
        return len(self.items) == 0
    
    def size(self):
        """
        Get number of items in stack.
        
        Returns:
            int: Number of items
        """
        # Return length of list
        return len(self.items)


class Queue:
    """
    A queue implementation using a list.
    
    Queue is FIFO (First In, First Out) - the first item added is the first
    item removed. Think of it like a line of people: people join at the back
    and leave from the front.
    
    We use a list where the front is index 0 and the back is the end.
    Note: dequeue() is O(n) because we remove from the front. For better
    performance, use collections.deque which has O(1) operations on both ends.
    """
    
    def __init__(self):
        """
        Initialize an empty queue.
        
        We use a list to store queue items. Index 0 is the front,
        and the end is the back.
        """
        # Initialize empty list to store queue items
        # Front is index 0, back is the end
        self.items = []
    
    def enqueue(self, item):
        """
        Add item to back of queue.
        
        Adds to the end of the list (back of queue). This is O(1) amortized.
        
        Args:
            item: Item to add
        """
        # Add item to end of list (back of queue)
        # append() adds to the end, which represents the back
        self.items.append(item)
    
    def dequeue(self):
        """
        Remove and return front item from queue.
        
        Removes from the beginning of the list (front of queue) and returns it.
        This is O(n) because removing from index 0 requires shifting all
        other elements. For better performance, use collections.deque.
        
        Returns:
            Item from front of queue
        
        Raises:
            IndexError: If queue is empty
        """
        # Remove and return first item (front of queue)
        # pop(0) removes from the beginning
        # This is O(n) - all remaining elements must shift left
        # For O(1) performance, use collections.deque.popleft()
        return self.items.pop(0)
    
    def peek(self):
        """
        Return front item without removing it.
        
        Just looks at the first element without modifying the queue.
        This is O(1).
        
        Returns:
            Item from front of queue
        
        Raises:
            IndexError: If queue is empty
        """
        # Return first item without removing it
        # [0] accesses the first element (front of queue)
        return self.items[0]
    
    def is_empty(self):
        """
        Check if queue is empty.
        
        Returns:
            bool: True if empty, False otherwise
        """
        # Check if list is empty
        return len(self.items) == 0
    
    def size(self):
        """
        Get number of items in queue.
        
        Returns:
            int: Number of items
        """
        # Return length of list
        return len(self.items)


def is_balanced_parentheses(expression):
    """
    Check if parentheses are balanced using a stack.
    
    This is a classic stack problem. The idea is:
    1. Push opening brackets onto stack
    2. When we see a closing bracket, check if it matches the top of stack
    3. If stack is empty at the end, parentheses are balanced
    
    Balanced examples: (), (()), ()(), ((()))
    Not balanced: ((), ))(, ())
    
    Time complexity: O(n) where n is length of expression
    Space complexity: O(n) in worst case (all opening brackets)
    
    Args:
        expression (str): String with parentheses
    
    Returns:
        bool: True if balanced, False otherwise
    
    Example:
        is_balanced_parentheses("()")      # True
        is_balanced_parentheses("(())")    # True
        is_balanced_parentheses("(()")     # False
        is_balanced_parentheses("())")     # False
    """
    # Create a stack to track opening brackets
    stack = Stack()
    
    # Define matching pairs
    # This makes it easy to check if brackets match
    pairs = {')': '(', '}': '{', ']': '['}
    
    # Iterate through each character in the expression
    for char in expression:
        # If it's an opening bracket, push onto stack
        if char in '({[':
            stack.push(char)
        
        # If it's a closing bracket
        elif char in ')}]':
            # Check if stack is empty (no matching opening bracket)
            if stack.is_empty():
                return False
            
            # Pop from stack and check if it matches
            # The opening bracket should match this closing bracket
            top = stack.pop()
            if top != pairs[char]:
                # Mismatch: opening and closing don't match
                return False
    
    # If stack is empty, all brackets were matched
    # If stack is not empty, there were unmatched opening brackets
    return stack.is_empty()

