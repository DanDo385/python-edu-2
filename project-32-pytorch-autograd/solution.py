"""
Project 32: PyTorch Autograd - SOLUTION

Complete solution demonstrating automatic differentiation.
"""

import torch


def compute_gradient_simple_function(x_value):
    """
    Compute gradient of f(x) = x^3 + 2x.
    
    Analytical: df/dx = 3x^2 + 2
    """
    # Create tensor with gradient tracking
    x = torch.tensor(x_value, requires_grad=True)
    
    # Perform computation
    y = x ** 3 + 2 * x
    
    # Compute gradient
    y.backward()
    
    # Return gradient
    return x.grad.item()


def compute_gradient_multivariate(a_value, b_value):
    """
    Compute gradients of z = 3a + 4b^2.
    
    Analytical: dz/da = 3, dz/db = 8b
    """
    # Create tensors with gradient tracking
    a = torch.tensor(a_value, requires_grad=True)
    b = torch.tensor(b_value, requires_grad=True)
    
    # Perform computation
    z = 3 * a + 4 * b ** 2
    
    # Compute gradients
    z.backward()
    
    # Return gradients
    return a.grad.item(), b.grad.item()


def verify_no_grad_context():
    """
    Verify that torch.no_grad() disables gradient tracking.
    """
    # With tracking enabled
    x1 = torch.tensor(2.0, requires_grad=True)
    y1 = x1 ** 2
    y1.backward()
    gradient_tracked = x1.grad.item()
    
    # With tracking disabled
    x2 = torch.tensor(2.0, requires_grad=True)
    with torch.no_grad():
        y2 = x2 ** 2
        # Cannot call backward() inside no_grad context
        # Or if we do, gradient won't be computed
        try:
            y2.backward()
            gradient_not_tracked = x2.grad
        except:
            gradient_not_tracked = None
    
    # Alternative: check that operations inside no_grad don't track
    x3 = torch.tensor(2.0, requires_grad=True)
    with torch.no_grad():
        y3 = x3 ** 2  # This won't track gradients
    # y3 doesn't have requires_grad, so backward() won't work
    gradient_not_tracked = None
    
    return gradient_tracked, gradient_not_tracked


def gradient_accumulation_demo():
    """
    Demonstrate that gradients accumulate if not zeroed.
    """
    x = torch.tensor(2.0, requires_grad=True)
    
    # First backward
    y1 = x ** 2
    y1.backward()
    gradient_after_one = x.grad.item()
    
    # Second backward (without zeroing)
    y2 = x ** 2
    y2.backward()  # Gradient accumulates!
    gradient_after_two = x.grad.item()
    
    return gradient_after_one, gradient_after_two


def zero_gradients_demo():
    """
    Demonstrate zeroing gradients to prevent accumulation.
    """
    x = torch.tensor(2.0, requires_grad=True)
    
    # Compute gradient
    y = x ** 2
    y.backward()
    gradient_before = x.grad.item()
    
    # Zero gradient
    x.grad.zero_()
    gradient_after = x.grad.item()
    
    return gradient_before, gradient_after


def compute_gradient_vector_output():
    """
    Compute gradient when output is a vector.
    """
    x = torch.tensor(1.0, requires_grad=True)
    
    # Vector output
    y = torch.stack([x ** 2, x ** 3])
    
    # For vector output, need to provide gradient vector
    # Usually sum the outputs or provide ones
    y.sum().backward()  # Sum then backward
    
    # Alternative: y.backward(torch.ones_like(y))
    
    return x.grad


def detach_tensor_demo():
    """
    Demonstrate detaching tensors from computation graph.
    """
    # Without detach
    x1 = torch.tensor(2.0, requires_grad=True)
    y1 = x1 ** 2
    z1 = y1 ** 2
    z1.backward()
    gradient_without_detach = x1.grad.item()
    
    # With detach
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = x2 ** 2
    y2_detached = y2.detach()  # Detach from graph
    z2 = y2_detached ** 2
    try:
        z2.backward()  # This won't work - y2_detached has no grad
        gradient_with_detach = x2.grad
    except:
        gradient_with_detach = None
    
    return gradient_with_detach, gradient_without_detach


# Example usage
if __name__ == "__main__":
    print("PyTorch Autograd Demo")
    print("=" * 50)
    
    # Simple function
    grad = compute_gradient_simple_function(2.0)
    print(f"Gradient of x^3 + 2x at x=2: {grad:.2f} (expected: 14.0)")
    
    # Multivariate
    grad_a, grad_b = compute_gradient_multivariate(1.0, 2.0)
    print(f"\nGradients of 3a + 4b^2:")
    print(f"  dz/da = {grad_a:.2f} (expected: 3.0)")
    print(f"  dz/db = {grad_b:.2f} (expected: 16.0)")
    
    # Accumulation
    grad_one, grad_two = gradient_accumulation_demo()
    print(f"\nGradient accumulation:")
    print(f"  After 1 backward: {grad_one:.2f}")
    print(f"  After 2 backward: {grad_two:.2f} (should be doubled)")
    
    print("\nAutograd makes gradient computation automatic! 🚀")
