"""
Project 32: PyTorch Autograd – Automatic Differentiation

Complete the functions below according to their docstrings.
Run pytest test.py -v to test your solutions.
"""

import torch
import numpy as np


def compute_gradient_simple_function(x_value):
    """
    Compute gradient of f(x) = x^3 + 2x at given x.
    
    Analytical solution: df/dx = 3x^2 + 2
    
    Args:
        x_value (float): Value at which to compute gradient
    
    Returns:
        float: Gradient value (dy/dx)
    
    Note:
        - Create tensor with requires_grad=True
        - Perform operations
        - Call .backward()
        - Access gradient via .grad
    """
    # TODO: Compute gradient using autograd
    # 1. Create tensor: torch.tensor(x_value, requires_grad=True)
    # 2. Compute y = x^3 + 2*x
    # 3. Call y.backward()
    # 4. Return x.grad
    return None


def compute_gradient_multivariate(a_value, b_value):
    """
    Compute gradients of z = 3a + 4b^2 with respect to a and b.
    
    Analytical solutions:
    - dz/da = 3
    - dz/db = 8b
    
    Args:
        a_value (float): Value for variable a
        b_value (float): Value for variable b
    
    Returns:
        tuple: (gradient_wrt_a, gradient_wrt_b)
    
    Note:
        - Both a and b need requires_grad=True
        - Call .backward() on z
        - Access gradients from a.grad and b.grad
    """
    # TODO: Compute gradients for multivariate function
    # 1. Create tensors for a and b with requires_grad=True
    # 2. Compute z = 3*a + 4*b**2
    # 3. Call z.backward()
    # 4. Return (a.grad, b.grad)
    return None, None


def verify_no_grad_context():
    """
    Verify that torch.no_grad() disables gradient tracking.
    
    Returns:
        tuple: (gradient_tracked, gradient_not_tracked)
            - gradient_tracked: Gradient computed with tracking enabled
            - gradient_not_tracked: Attempt to compute gradient with tracking disabled (should be None)
    
    Note:
        - First compute gradient normally (should work)
        - Then compute gradient inside torch.no_grad() context (should be None)
    """
    # TODO: Demonstrate no_grad context
    # 1. Compute gradient normally (with tracking)
    # 2. Try to compute gradient inside torch.no_grad() context
    #    (gradient should be None or not computed)
    gradient_tracked = None
    gradient_not_tracked = None
    return gradient_tracked, gradient_not_tracked


def gradient_accumulation_demo():
    """
    Demonstrate that gradients accumulate if not zeroed.
    
    Returns:
        tuple: (gradient_after_one_backward, gradient_after_two_backward)
            - Shows that gradients accumulate when .backward() is called multiple times
    
    Note:
        - Call .backward() once, check gradient
        - Call .backward() again, check gradient (should be doubled)
    """
    # TODO: Show gradient accumulation
    # 1. Create tensor with requires_grad=True
    # 2. Compute y1, call backward(), store gradient
    # 3. Compute y2 (same operation), call backward() again
    # 4. Check that gradient doubled
    gradient_after_one = None
    gradient_after_two = None
    return gradient_after_one, gradient_after_two


def zero_gradients_demo():
    """
    Demonstrate zeroing gradients to prevent accumulation.
    
    Returns:
        tuple: (gradient_before_zero, gradient_after_zero)
            - Shows that .grad.zero_() resets gradients
    
    Note:
        - Compute gradient
        - Zero gradient using .grad.zero_()
        - Verify gradient is zero
    """
    # TODO: Show gradient zeroing
    # 1. Create tensor, compute gradient
    # 2. Store gradient value
    # 3. Call .grad.zero_() to zero gradient
    # 4. Verify gradient is zero
    gradient_before = None
    gradient_after = None
    return gradient_before, gradient_after


def compute_gradient_vector_output():
    """
    Compute gradient when output is a vector (not scalar).
    
    For vector outputs, need to provide gradient vector to backward().
    
    Args:
        None
    
    Returns:
        torch.Tensor: Gradient of sum of outputs
    
    Note:
        - Create tensor with requires_grad=True
        - Compute vector output (e.g., [x^2, x^3])
        - Call backward() with gradient vector (usually ones)
        # backward(torch.ones_like(output))
    """
    # TODO: Compute gradient for vector output
    # 1. Create tensor x with requires_grad=True
    # 2. Compute y = [x^2, x^3] (vector output)
    # 3. Call y.backward(torch.ones_like(y)) or y.sum().backward()
    # 4. Return x.grad
    return None


def detach_tensor_demo():
    """
    Demonstrate detaching tensors from computation graph.
    
    Returns:
        tuple: (gradient_with_detach, gradient_without_detach)
            - Shows that .detach() stops gradient flow
    
    Note:
        - Create computation with detach in middle
        - Compare gradients with and without detach
    """
    # TODO: Show tensor detachment
    # 1. Create computation: x -> y -> z (with y.detach())
    # 2. Compute gradient of z w.r.t. x (should be None or 0)
    # 3. Compare with computation without detach
    gradient_with_detach = None
    gradient_without_detach = None
    return gradient_with_detach, gradient_without_detach
