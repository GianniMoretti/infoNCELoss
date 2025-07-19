#!/usr/bin/env python3
"""
Test script for the new InfoNCE CUDA implementation
Exactly replicates the provided Python code and compares results
"""

import torch
import torch.nn.functional as F
from cublaze.infonce import InfoNCELoss

def info_nce_loss_reference(features, temperature=0.5):
    """
    Reference implementation in pure PyTorch (from user provided code)
    """
    device = features.device
    batch_size = features.shape[0] // 2

    # Compute similarity matrix (dot product)
    similarity_matrix = torch.matmul(features, features.T)  # (2B, 2B)

    # Remove self-similarity by masking the diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Labels: positive pair for i is at (i + B) % (2B)
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels])

    # Scale similarities by temperature
    similarity_matrix /= temperature

    # Apply cross entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def test_correctness():
    """
    Correctness test comparing with reference implementation
    """
    print("=== Correctness Test ===")
    
    # Test parameters
    batch_size = 4  # So 2*batch_size = 8
    feature_dim = 128
    temperature = 0.5
    
    # Generate random features
    torch.manual_seed(42)
    features = torch.randn(2 * batch_size, feature_dim, device='cuda', requires_grad=True)
    features = F.normalize(features, dim=1)  # L2 normalize
    
    # Test CUDA implementation
    print("Testing CUDA implementation...")
    features_cuda = features.clone().detach().requires_grad_(True)
    cuda_loss_fn = InfoNCELoss(temperature=temperature)
    loss_cuda = cuda_loss_fn(features_cuda)
    
    # Test reference implementation
    print("Testing reference implementation...")
    features_ref = features.clone().detach().requires_grad_(True)
    loss_ref = info_nce_loss_reference(features_ref, temperature)
    
    print(f"CUDA Loss: {loss_cuda.item():.6f}")
    print(f"Reference Loss: {loss_ref.item():.6f}")
    print(f"Difference: {abs(loss_cuda.item() - loss_ref.item()):.8f}")
    
    # Test dei gradienti
    print("\nTesting gradients...")
    loss_cuda.backward()
    loss_ref.backward()
    
    grad_diff = torch.abs(features_cuda.grad - features_ref.grad)
    max_grad_diff = grad_diff.max().item()
    mean_grad_diff = grad_diff.mean().item()
    
    print(f"Max gradient difference: {max_grad_diff:.8f}")
    print(f"Mean gradient difference: {mean_grad_diff:.8f}")
    
    # Check di tolleranza
    loss_tol = 1e-5
    grad_tol = 1e-4
    
    loss_match = abs(loss_cuda.item() - loss_ref.item()) < loss_tol
    grad_match = max_grad_diff < grad_tol
    
    print(f"\nLoss match (tol={loss_tol}): {'✓' if loss_match else '✗'}")
    print(f"Gradient match (tol={grad_tol}): {'✓' if grad_match else '✗'}")
    
    return loss_match and grad_match

def test_performance():
    """
    Performance test comparing execution times
    """
    print("\n=== Performance Test ===")
    
    batch_size = 1024  # 2*batch_size = 2048
    feature_dim = 512
    temperature = 0.5
    num_iterations = 10
    
    # Generate random features
    features = torch.randn(2 * batch_size, feature_dim, device='cuda')
    cuda_loss_fn = InfoNCELoss(temperature=temperature)
    
    # Warm-up
    for _ in range(10):
        f_cuda = features.clone().requires_grad_(True)
        loss_cuda = cuda_loss_fn(f_cuda)
        loss_cuda.backward()
        
        f_ref = features.clone().requires_grad_(True)
        loss_ref = info_nce_loss_reference(f_ref, temperature)
        loss_ref.backward()
    
    torch.cuda.synchronize()
    
    # Test CUDA implementation - Forward pass
    import time
    start_time = time.time()
    for _ in range(num_iterations):
        f_cuda = features.clone().requires_grad_(True)
        loss_cuda = cuda_loss_fn(f_cuda)
    torch.cuda.synchronize()
    cuda_forward_time = time.time() - start_time
    
    # Test CUDA implementation - Forward + Backward pass
    start_time = time.time()
    for _ in range(num_iterations):
        f_cuda = features.clone().requires_grad_(True)
        loss_cuda = cuda_loss_fn(f_cuda)
        loss_cuda.backward()
    torch.cuda.synchronize()
    cuda_total_time = time.time() - start_time
    cuda_backward_time = cuda_total_time - cuda_forward_time
    
    # Test reference implementation - Forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        f_ref = features.clone().requires_grad_(True)
        loss_ref = info_nce_loss_reference(f_ref, temperature)
    torch.cuda.synchronize()
    ref_forward_time = time.time() - start_time
    
    # Test reference implementation - Forward + Backward pass
    start_time = time.time()
    for _ in range(num_iterations):
        f_ref = features.clone().requires_grad_(True)
        loss_ref = info_nce_loss_reference(f_ref, temperature)
        loss_ref.backward()
    torch.cuda.synchronize()
    ref_total_time = time.time() - start_time
    ref_backward_time = ref_total_time - ref_forward_time
    
    print(f"CUDA forward time: {cuda_forward_time:.4f}s ({cuda_forward_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"CUDA backward time: {cuda_backward_time:.4f}s ({cuda_backward_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"CUDA total time: {cuda_total_time:.4f}s ({cuda_total_time/num_iterations*1000:.2f}ms per iteration)")
    
    print(f"Reference forward time: {ref_forward_time:.4f}s ({ref_forward_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Reference backward time: {ref_backward_time:.4f}s ({ref_backward_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Reference total time: {ref_total_time:.4f}s ({ref_total_time/num_iterations*1000:.2f}ms per iteration)")
    
    print(f"Forward speedup: {ref_forward_time/cuda_forward_time:.2f}x")
    print(f"Backward speedup: {ref_backward_time/cuda_backward_time:.2f}x")
    print(f"Total speedup: {ref_total_time/cuda_total_time:.2f}x")

def test_different_sizes():
    """
    Test with different batch sizes
    """
    print("\n=== Test with Different Sizes ===")
    
    temperature = 0.5
    feature_dim = 256
    cuda_loss_fn = InfoNCELoss(temperature=temperature)
    
    batch_sizes = [128, 256, 512, 2048]  # These are B values, so 2*B will be actual batch
    
    for B in batch_sizes:
        print(f"\nTesting batch_size = {B} (total samples = {2*B})")
        
        features = torch.randn(2 * B, feature_dim, device='cuda', requires_grad=True)
        
        try:
            loss_cuda = cuda_loss_fn(features)
            loss_ref = info_nce_loss_reference(features, temperature)
            
            diff = abs(loss_cuda.item() - loss_ref.item())
            print(f"  Loss difference: {diff:.8f}")
            
            if diff < 1e-5:
                print("  ✓ Passed")
            else:
                print("  ✗ Failed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")


print("Testing New InfoNCE CUDA Implementation")
print("="*50)

# Verifica che CUDA sia disponibile
if not torch.cuda.is_available():
    print("CUDA not available! Exiting...")
    exit(1)

print(f"Using device: {torch.cuda.get_device_name()}")
print(f"PyTorch version: {torch.__version__}")

try:
    # Test di correttezza
    correctness_passed = test_correctness()
    
    if correctness_passed:
        print("\n🎉 Correctness tests PASSED!")
        
        # Test di performance solo se la correttezza è OK
        test_performance()
        
        # Test con diverse dimensioni
        test_different_sizes()
        
    else:
        print("\n❌ Correctness tests FAILED!")
        
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
