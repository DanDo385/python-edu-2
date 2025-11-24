"""Test suite for Project 12: Linked List"""
import pytest
from exercise import Node, SinglyLinkedList

def test_node():
    node = Node(5)
    assert node.data == 5
    assert node.next is None

def test_append():
    ll = SinglyLinkedList()
    ll.append(1)
    ll.append(2)
    assert ll.to_list() == [1, 2]

def test_prepend():
    ll = SinglyLinkedList()
    ll.prepend(2)
    ll.prepend(1)
    assert ll.to_list() == [1, 2]

def test_find():
    ll = SinglyLinkedList()
    ll.append(1)
    ll.append(2)
    node = ll.find(2)
    assert node is not None
    assert node.data == 2

def test_remove():
    ll = SinglyLinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    assert ll.remove(2) == True
    assert ll.to_list() == [1, 3]

def test_reverse():
    ll = SinglyLinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    ll.reverse()
    assert ll.to_list() == [3, 2, 1]

