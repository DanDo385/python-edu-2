"""Test suite for Project 11: Stack and Queue"""
import pytest
from exercise import Stack, Queue, is_balanced_parentheses

def test_stack():
    stack = Stack()
    assert stack.is_empty()
    stack.push(1)
    stack.push(2)
    assert stack.size() == 2
    assert stack.peek() == 2
    assert stack.pop() == 2
    assert stack.pop() == 1
    assert stack.is_empty()

def test_queue():
    queue = Queue()
    assert queue.is_empty()
    queue.enqueue(1)
    queue.enqueue(2)
    assert queue.size() == 2
    assert queue.peek() == 1
    assert queue.dequeue() == 1
    assert queue.dequeue() == 2
    assert queue.is_empty()

def test_balanced_parentheses():
    assert is_balanced_parentheses("()") == True
    assert is_balanced_parentheses("(())") == True
    assert is_balanced_parentheses("(()") == False
    assert is_balanced_parentheses("())") == False

