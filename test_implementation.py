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

def test_correctness_single(use_cublas, version_name):
    """
    Test correctness for a single implementation version
    """
    print(f"\n--- Testing {version_name} ---")
    
    # Test parameters
    batch_size = 512
    feature_dim = 128
    temperature = 0.5
    tolerance = 1e-4
    
    # Generate random features
    torch.manual_seed(42)
    features = torch.randn(2 * batch_size, feature_dim, device='cuda', requires_grad=True)
    features = F.normalize(features, dim=1)  # L2 normalize
    
    try:
        # Test CUDA implementation
        print(f"Testing {version_name} implementation...")
        features_cuda = features.clone().detach().requires_grad_(True)
        cuda_loss_fn = InfoNCELoss(temperature=temperature, use_cublas=use_cublas)
        loss_cuda = cuda_loss_fn(features_cuda)
        
        # Test reference implementation
        print("Testing reference implementation...")
        features_ref = features.clone().detach().requires_grad_(True)
        loss_ref = info_nce_loss_reference(features_ref, temperature)
        
        print(f"{version_name} Loss: {loss_cuda.item():.6f}")
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
        loss_match = abs(loss_cuda.item() - loss_ref.item()) < tolerance
        grad_match = max_grad_diff < tolerance
        
        print(f"\nUsing tolerance: {tolerance}")
        print(f"Loss match: {'✓' if loss_match else '✗'}")
        print(f"Gradient match: {'✓' if grad_match else '✗'}")
        
        result = loss_match and grad_match
        print(f"{version_name} correctness test: {'✓ PASSED' if result else '✗ FAILED'}")
        return result
        
    except Exception as e:
        print(f"✗ {version_name} test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_correctness():
    """
    Correctness test comparing both implementations with reference
    """
    print("\n=== Correctness Test ===\n")
    
    results = {}
    
    # Test Custom Kernels version (CUBLAS=False)
    results['Custom Kernels'] = test_correctness_single(use_cublas=False, version_name="Custom Kernels")
    
    # Test CUBLAS version (CUBLAS=True)  
    results['CUBLAS'] = test_correctness_single(use_cublas=True, version_name="CUBLAS")
    
    print(f"\n=== Correctness Summary ===")
    for version, passed in results.items():
        print(f"{version}: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return results

def test_performance():
    """
    Performance test comparing execution times for both implementations
    """
    print("\n=== Performance Test ===")
    
    batch_size = 1024  # 2*batch_size = 2048
    print(f"\nTesting with batch_size = {batch_size} (total samples = {2 * batch_size})\n")
    feature_dim = 512
    temperature = 0.5
    num_iterations = 10
    
    # Generate random features
    features = torch.randn(2 * batch_size, feature_dim, device='cuda')
    
    # Create loss functions for both implementations
    custom_loss_fn = InfoNCELoss(temperature=temperature, use_cublas=False)
    cublas_loss_fn = InfoNCELoss(temperature=temperature, use_cublas=True)
    
    # Warm-up for all implementations
    print("Warming up...")
    for _ in range(5):
        f_custom = features.clone().requires_grad_(True)
        loss_custom = custom_loss_fn(f_custom)
        loss_custom.backward()
        
        f_cublas = features.clone().requires_grad_(True)
        loss_cublas = cublas_loss_fn(f_cublas)
        loss_cublas.backward()
        
        f_ref = features.clone().requires_grad_(True)
        loss_ref = info_nce_loss_reference(f_ref, temperature)
        loss_ref.backward()
    
    torch.cuda.synchronize()
    
    def benchmark_implementation(loss_fn, name):
        """Helper function to benchmark an implementation"""
        import time
        
        # Forward pass timing
        start_time = time.time()
        for _ in range(num_iterations):
            f = features.clone().requires_grad_(True)
            loss = loss_fn(f) if callable(loss_fn) else loss_fn(f, temperature)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        # Forward + Backward pass timing
        start_time = time.time()
        for _ in range(num_iterations):
            f = features.clone().requires_grad_(True)
            loss = loss_fn(f) if callable(loss_fn) else loss_fn(f, temperature)
            loss.backward()
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        backward_time = total_time - forward_time
        
        print(f"{name} performance:")
        print(f"  Forward time: {forward_time:.4f}s ({forward_time/num_iterations*1000:.2f}ms per iteration)")
        print(f"  Backward time: {backward_time:.4f}s ({backward_time/num_iterations*1000:.2f}ms per iteration)")
        print(f"  Total time: {total_time:.4f}s ({total_time/num_iterations*1000:.2f}ms per iteration)")
        print()
        
        return forward_time, backward_time, total_time
    
    # Benchmark all implementations
    print("Benchmarking implementations...\n")
    
    try:
        custom_forward, custom_backward, custom_total = benchmark_implementation(custom_loss_fn, "Custom Kernels")
    except Exception as e:
        print(f"Custom Kernels benchmark failed: {e}")
        custom_forward = custom_backward = custom_total = float('inf')
        
    try:
        cublas_forward, cublas_backward, cublas_total = benchmark_implementation(cublas_loss_fn, "CUBLAS")
    except Exception as e:
        print(f"CUBLAS benchmark failed: {e}")
        cublas_forward = cublas_backward = cublas_total = float('inf')
    
    try:
        ref_forward, ref_backward, ref_total = benchmark_implementation(info_nce_loss_reference, "PyTorch Reference")
    except Exception as e:
        print(f"PyTorch Reference benchmark failed: {e}")
        ref_forward = ref_backward = ref_total = float('inf')
    
    # Speedup comparison
    print("=== SPEEDUP COMPARISON ===\n")
    
    if ref_total != float('inf'):
        if custom_total != float('inf'):
            print(f"Custom Kernels vs PyTorch:")
            print(f"  Forward speedup: {ref_forward/custom_forward:.2f}x")
            print(f"  Backward speedup: {ref_backward/custom_backward:.2f}x")
            print(f"  Total speedup: {ref_total/custom_total:.2f}x")
            print()
        
        if cublas_total != float('inf'):
            print(f"CUBLAS vs PyTorch:")
            print(f"  Forward speedup: {ref_forward/cublas_forward:.2f}x")
            print(f"  Backward speedup: {ref_backward/cublas_backward:.2f}x")
            print(f"  Total speedup: {ref_total/cublas_total:.2f}x")
            print()
    
    if custom_total != float('inf') and cublas_total != float('inf'):
        print(f"CUBLAS vs Custom Kernels:")
        print(f"  Forward speedup: {custom_forward/cublas_forward:.2f}x")
        print(f"  Backward speedup: {custom_backward/cublas_backward:.2f}x")
        print(f"  Total speedup: {custom_total/cublas_total:.2f}x")

def test_different_sizes():
    """
    Test with different batch sizes for both implementations
    """
    print("\n=== Test with Different Sizes ===")
    
    temperature = 0.5
    feature_dim = 256
    tolerance = 1e-4
    
    batch_sizes = [128, 256, 512, 1024]  # These are B values, so 2*B will be actual batch
    implementations = [
        (False, "Custom Kernels"),
        (True, "CUBLAS")
    ]
    
    results = {}
    
    for use_cublas, impl_name in implementations:
        print(f"\n--- Testing {impl_name} Implementation ---")
        results[impl_name] = {}
        
        cuda_loss_fn = InfoNCELoss(temperature=temperature, use_cublas=use_cublas)
        
        for B in batch_sizes:
            print(f"\nTesting batch_size = {B} (total samples = {2*B})")
            print(f"  Using tolerance: {tolerance}")
            
            # Generate features for both tests
            features = torch.randn(2 * B, feature_dim, device='cuda')
            
            try:
                # Test CUDA implementation
                features_cuda = features.clone().requires_grad_(True)
                loss_cuda = cuda_loss_fn(features_cuda)
                
                # Test reference implementation  
                features_ref = features.clone().requires_grad_(True)
                loss_ref = info_nce_loss_reference(features_ref, temperature)
                
                # Loss comparison
                loss_diff = abs(loss_cuda.item() - loss_ref.item())
                print(f"  Loss difference: {loss_diff:.8f}")
                
                # Gradient comparison
                print("  Testing gradients...")
                loss_cuda.backward()
                loss_ref.backward()
                
                grad_diff = torch.abs(features_cuda.grad - features_ref.grad)
                max_grad_diff = grad_diff.max().item()
                mean_grad_diff = grad_diff.mean().item()
                
                print(f"  Max gradient difference: {max_grad_diff:.8f}")
                print(f"  Mean gradient difference: {mean_grad_diff:.8f}")
                
                # Check tolerances
                loss_match = loss_diff < tolerance
                grad_match = max_grad_diff < tolerance
                
                if loss_match and grad_match:
                    print("  ✓ Passed (both loss and gradients)")
                    results[impl_name][B] = "PASSED"
                elif loss_match:
                    print(f"  ⚠ Loss passed, gradients failed (max grad diff {max_grad_diff:.2e} > tol {tolerance})")
                    results[impl_name][B] = "PARTIAL"
                elif grad_match:
                    print(f"  ⚠ Gradients passed, loss failed (loss diff {loss_diff:.2e} > tol {tolerance})")
                    results[impl_name][B] = "PARTIAL"
                else:
                    print(f"  ✗ Failed (loss diff {loss_diff:.2e}, max grad diff {max_grad_diff:.2e} > tol {tolerance})")
                    results[impl_name][B] = "FAILED"
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[impl_name][B] = "ERROR"
    
    # Summary
    print(f"\n=== Different Sizes Test Summary ===")
    for impl_name in results:
        print(f"\n{impl_name}:")
        for B, status in results[impl_name].items():
            status_symbol = {"PASSED": "✓", "PARTIAL": "⚠", "FAILED": "✗", "ERROR": "✗"}[status]
            print(f"  Batch size {B}: {status_symbol} {status}")
    
    return results

print("Testing New InfoNCE CUDA Implementation")
print("="*50)

# Verifica che CUDA sia disponibile
if not torch.cuda.is_available():
    print("CUDA not available! Exiting...")
    exit(1)

print(f"Using device: {torch.cuda.get_device_name()}")
print(f"PyTorch version: {torch.__version__}")

# Track overall results
all_tests_passed = True
test_results = {}

try:
    # Test di correttezza per entrambe le implementazioni
    print("Running correctness tests for both implementations...")
    correctness_results = test_correctness()
    test_results['correctness'] = correctness_results
    
    # Continue with other tests regardless of correctness results
    print("Running tests with different sizes...")
    try:
        size_results = test_different_sizes()
        test_results['different_sizes'] = size_results
    except Exception as e:
        print(f"Different sizes test failed: {e}")
        test_results['different_sizes'] = "ERROR"
    
    # Test di performance
    print("Running performance tests...")
    try:
        test_performance()
        test_results['performance'] = "COMPLETED"
    except Exception as e:
        print(f"Performance test failed: {e}")
        test_results['performance'] = "ERROR"
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    
    print("\nCorrectness Tests:")
    if isinstance(test_results.get('correctness'), dict):
        for version, passed in test_results['correctness'].items():
            print(f"  {version}: {'✓ PASSED' if passed else '✗ FAILED'}")
            if not passed:
                all_tests_passed = False
    else:
        print("  ERROR in correctness tests")
        all_tests_passed = False
    
    print(f"\nDifferent Sizes Tests: {'✓ COMPLETED' if test_results.get('different_sizes') != 'ERROR' else '✗ ERROR'}")
    if test_results.get('different_sizes') == 'ERROR':
        all_tests_passed = False
    
    print(f"Performance Tests: {'✓ COMPLETED' if test_results.get('performance') == 'COMPLETED' else '✗ ERROR'}")
    
    print(f"\nOverall Result: {'✓ SUCCESS' if correctness_results.get('CUBLAS', False) or correctness_results.get('Custom Kernels', False) else '✗ SOME ISSUES FOUND'}")
    
    if correctness_results.get('CUBLAS', False) and correctness_results.get('Custom Kernels', False):
        print("\n🎉 Both implementations are working correctly!")
    elif correctness_results.get('CUBLAS', False):
        print("\n⚠️  CUBLAS implementation is working, but Custom Kernels need attention")
    elif correctness_results.get('Custom Kernels', False):
        print("\n⚠️  Custom Kernels implementation is working, but CUBLAS needs attention")
    else:
        print("\n⚠️  Both implementations need attention - check the error logs above")
        
except Exception as e:
    print(f"\nTest suite failed with error: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

print(f"\nFinal Status: {'SUCCESS' if any(test_results.get('correctness', {}).values()) else 'NEEDS_ATTENTION'}")