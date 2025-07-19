#!/usr/bin/env python3
"""
Performance analysis script for InfoNCE CUDA implementations
Tests performance across different batch sizes and generates visualization plots
"""

import torch
import torch.nn.functional as F
from cublaze.infonce import InfoNCELoss
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def info_nce_loss_reference(features, temperature=0.5):
    """
    Reference implementation in pure PyTorch
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

def benchmark_implementation(loss_fn, features, num_iterations=10, name="Implementation"):
    """
    Benchmark a single implementation
    """
    # Warm-up
    for _ in range(5):
        f = features.clone().requires_grad_(True)
        if callable(loss_fn):
            loss = loss_fn(f)
        else:
            loss = loss_fn(f, 0.5)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Forward pass timing
    forward_times = []
    for _ in range(num_iterations):
        f = features.clone().requires_grad_(True)
        start = time.time()
        if callable(loss_fn):
            loss = loss_fn(f)
        else:
            loss = loss_fn(f, 0.5)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start)
    
    # Backward pass timing
    backward_times = []
    for _ in range(num_iterations):
        f = features.clone().requires_grad_(True)
        if callable(loss_fn):
            loss = loss_fn(f)
        else:
            loss = loss_fn(f, 0.5)
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start)
    
    forward_time = np.mean(forward_times)
    backward_time = np.mean(backward_times)
    total_time = forward_time + backward_time
    
    return forward_time, backward_time, total_time

def test_correctness(features, temperature=0.5):
    """
    Test correctness of implementations vs reference
    """
    # Reference implementation
    features_ref = features.clone().detach().requires_grad_(True)
    loss_ref = info_nce_loss_reference(features_ref, temperature)
    loss_ref.backward()
    
    # Custom kernels implementation
    features_custom = features.clone().detach().requires_grad_(True)
    loss_fn_custom = InfoNCELoss(temperature=temperature, use_cublas=False)
    loss_custom = loss_fn_custom(features_custom)
    loss_custom.backward()
    
    # CUBLAS implementation
    features_cublas = features.clone().detach().requires_grad_(True)
    loss_fn_cublas = InfoNCELoss(temperature=temperature, use_cublas=True)
    loss_cublas = loss_fn_cublas(features_cublas)
    loss_cublas.backward()
    
    # Calculate errors
    loss_error_custom = abs(loss_custom.item() - loss_ref.item())
    loss_error_cublas = abs(loss_cublas.item() - loss_ref.item())
    
    grad_error_custom = (features_custom.grad - features_ref.grad).abs().max().item()
    grad_error_cublas = (features_cublas.grad - features_ref.grad).abs().max().item()
    
    return {
        'loss_error_custom': loss_error_custom,
        'loss_error_cublas': loss_error_cublas,
        'grad_error_custom': grad_error_custom,
        'grad_error_cublas': grad_error_cublas
    }

def run_performance_analysis():
    """
    Run comprehensive performance analysis
    """
    print("InfoNCE CUDA Performance Analysis")
    print("="*50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available! Exiting...")
        return
    
    print(f"Using device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Test parameters
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # B values (actual batch will be 2*B)
    feature_dim = 256
    temperature = 0.5
    num_iterations = 20
    
    # Storage for results
    results = {
        'batch_sizes': [],
        'forward_times_ref': [],
        'backward_times_ref': [],
        'total_times_ref': [],
        'forward_times_custom': [],
        'backward_times_custom': [],
        'total_times_custom': [],
        'forward_times_cublas': [],
        'backward_times_cublas': [],
        'total_times_cublas': [],
        'loss_errors_custom': [],
        'loss_errors_cublas': [],
        'grad_errors_custom': [],
        'grad_errors_cublas': []
    }
    
    # Run tests for each batch size
    for B in batch_sizes:
        actual_batch_size = 2 * B
        print(f"\nTesting batch size {B} (total samples: {actual_batch_size})")
        
        # Generate features
        features = torch.randn(actual_batch_size, feature_dim, device='cuda')
        features = F.normalize(features, dim=1)
        
        try:
            # Test correctness
            correctness = test_correctness(features, temperature)
            
            # Benchmark PyTorch reference
            print("  Benchmarking PyTorch reference...")
            ref_forward, ref_backward, ref_total = benchmark_implementation(
                info_nce_loss_reference, features, num_iterations, "Reference"
            )
            
            # Benchmark custom kernels
            print("  Benchmarking custom kernels...")
            custom_loss_fn = InfoNCELoss(temperature=temperature, use_cublas=False)
            custom_forward, custom_backward, custom_total = benchmark_implementation(
                custom_loss_fn, features, num_iterations, "Custom"
            )
            
            # Benchmark CUBLAS
            print("  Benchmarking CUBLAS...")
            cublas_loss_fn = InfoNCELoss(temperature=temperature, use_cublas=True)
            cublas_forward, cublas_backward, cublas_total = benchmark_implementation(
                cublas_loss_fn, features, num_iterations, "CUBLAS"
            )
            
            # Store results
            results['batch_sizes'].append(B)
            results['forward_times_ref'].append(ref_forward * 1000)  # Convert to ms
            results['backward_times_ref'].append(ref_backward * 1000)
            results['total_times_ref'].append(ref_total * 1000)
            results['forward_times_custom'].append(custom_forward * 1000)
            results['backward_times_custom'].append(custom_backward * 1000)
            results['total_times_custom'].append(custom_total * 1000)
            results['forward_times_cublas'].append(cublas_forward * 1000)
            results['backward_times_cublas'].append(cublas_backward * 1000)
            results['total_times_cublas'].append(cublas_total * 1000)
            results['loss_errors_custom'].append(correctness['loss_error_custom'])
            results['loss_errors_cublas'].append(correctness['loss_error_cublas'])
            results['grad_errors_custom'].append(correctness['grad_error_custom'])
            results['grad_errors_cublas'].append(correctness['grad_error_cublas'])
            
            print(f"    Reference: {ref_total*1000:.2f}ms")
            print(f"    Custom: {custom_total*1000:.2f}ms")
            print(f"    CUBLAS: {cublas_total*1000:.2f}ms")
            
        except Exception as e:
            print(f"    Error for batch size {B}: {e}")
    
    # Generate plots
    generate_plots(results)
    
    print(f"\nPerformance analysis complete! Plots saved in 'images/' directory.")

def generate_plots(results):
    """
    Generate all visualization plots
    """
    batch_sizes = results['batch_sizes']
    actual_batch_sizes = [2 * b for b in batch_sizes]  # Convert to actual batch sizes (2*B)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # 1. Execution Times Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Forward times
    ax1.loglog(actual_batch_sizes, results['forward_times_ref'], 'o-', label='PyTorch Reference', linewidth=2, markersize=8)
    ax1.loglog(actual_batch_sizes, results['forward_times_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax1.loglog(actual_batch_sizes, results['forward_times_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax1.set_xlabel('Actual Batch Size (2*B)')
    ax1.set_ylabel('Forward Time (ms)')
    ax1.set_title('Forward Pass Execution Times')
    ax1.set_xticks(actual_batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in actual_batch_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Backward times
    ax2.loglog(actual_batch_sizes, results['backward_times_ref'], 'o-', label='PyTorch Reference', linewidth=2, markersize=8)
    ax2.loglog(actual_batch_sizes, results['backward_times_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax2.loglog(actual_batch_sizes, results['backward_times_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax2.set_xlabel('Actual Batch Size (2*B)')
    ax2.set_ylabel('Backward Time (ms)')
    ax2.set_title('Backward Pass Execution Times')
    ax2.set_xticks(actual_batch_sizes)
    ax2.set_xticklabels([str(bs) for bs in actual_batch_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Total times
    ax3.loglog(actual_batch_sizes, results['total_times_ref'], 'o-', label='PyTorch Reference', linewidth=2, markersize=8)
    ax3.loglog(actual_batch_sizes, results['total_times_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax3.loglog(actual_batch_sizes, results['total_times_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax3.set_xlabel('Actual Batch Size (2*B)')
    ax3.set_ylabel('Total Time (ms)')
    ax3.set_title('Total Execution Times')
    ax3.set_xticks(actual_batch_sizes)
    ax3.set_xticklabels([str(bs) for bs in actual_batch_sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/execution_times.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Speedup Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate speedups relative to PyTorch reference
    custom_forward_speedup = [ref/custom for ref, custom in zip(results['forward_times_ref'], results['forward_times_custom'])]
    cublas_forward_speedup = [ref/cublas for ref, cublas in zip(results['forward_times_ref'], results['forward_times_cublas'])]
    
    custom_backward_speedup = [ref/custom for ref, custom in zip(results['backward_times_ref'], results['backward_times_custom'])]
    cublas_backward_speedup = [ref/cublas for ref, cublas in zip(results['backward_times_ref'], results['backward_times_cublas'])]
    
    custom_total_speedup = [ref/custom for ref, custom in zip(results['total_times_ref'], results['total_times_custom'])]
    cublas_total_speedup = [ref/cublas for ref, cublas in zip(results['total_times_ref'], results['total_times_cublas'])]
    
    # Forward speedup
    ax1.semilogx(actual_batch_sizes, custom_forward_speedup, 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax1.semilogx(actual_batch_sizes, cublas_forward_speedup, '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Reference (1x)')
    ax1.set_xlabel('Actual Batch Size (2*B)')
    ax1.set_ylabel('Speedup vs PyTorch')
    ax1.set_title('Forward Pass Speedup')
    ax1.set_xticks(actual_batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in actual_batch_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Backward speedup
    ax2.semilogx(actual_batch_sizes, custom_backward_speedup, 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax2.semilogx(actual_batch_sizes, cublas_backward_speedup, '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Reference (1x)')
    ax2.set_xlabel('Actual Batch Size (2*B)')
    ax2.set_ylabel('Speedup vs PyTorch')
    ax2.set_title('Backward Pass Speedup')
    ax2.set_xticks(actual_batch_sizes)
    ax2.set_xticklabels([str(bs) for bs in actual_batch_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Total speedup
    ax3.semilogx(actual_batch_sizes, custom_total_speedup, 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax3.semilogx(actual_batch_sizes, cublas_total_speedup, '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Reference (1x)')
    ax3.set_xlabel('Actual Batch Size (2*B)')
    ax3.set_ylabel('Speedup vs PyTorch')
    ax3.set_title('Total Speedup')
    ax3.set_xticks(actual_batch_sizes)
    ax3.set_xticklabels([str(bs) for bs in actual_batch_sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Loss Error Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.semilogy(actual_batch_sizes, results['loss_errors_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax.semilogy(actual_batch_sizes, results['loss_errors_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax.set_xlabel('Actual Batch Size (2*B)')
    ax.set_ylabel('Loss Error vs Reference')
    ax.set_title('Loss Computation Error')
    # Don't show specific batch size values on x-axis
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/loss_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gradient Error Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.semilogy(actual_batch_sizes, results['grad_errors_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax.semilogy(actual_batch_sizes, results['grad_errors_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax.set_xlabel('Actual Batch Size (2*B)')
    ax.set_ylabel('Max Gradient Error vs Reference')
    ax.set_title('Gradient Computation Error')
    # Don't show specific batch size values on x-axis
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/gradient_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Combined Performance Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total times
    ax1.loglog(actual_batch_sizes, results['total_times_ref'], 'o-', label='PyTorch Reference', linewidth=2, markersize=8)
    ax1.loglog(actual_batch_sizes, results['total_times_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax1.loglog(actual_batch_sizes, results['total_times_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax1.set_xlabel('Actual Batch Size (2*B)')
    ax1.set_ylabel('Total Time (ms)')
    ax1.set_title('Total Execution Times')
    ax1.set_xticks(actual_batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in actual_batch_sizes], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Total speedup
    ax2.semilogx(actual_batch_sizes, custom_total_speedup, 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax2.semilogx(actual_batch_sizes, cublas_total_speedup, '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Reference (1x)')
    ax2.set_xlabel('Actual Batch Size (2*B)')
    ax2.set_ylabel('Speedup vs PyTorch')
    ax2.set_title('Total Speedup')
    ax2.set_xticks(actual_batch_sizes)
    ax2.set_xticklabels([str(bs) for bs in actual_batch_sizes], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss error - don't show specific batch size values
    ax3.semilogy(actual_batch_sizes, results['loss_errors_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax3.semilogy(actual_batch_sizes, results['loss_errors_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax3.set_xlabel('Actual Batch Size (2*B)')
    ax3.set_ylabel('Loss Error vs Reference')
    ax3.set_title('Loss Computation Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gradient error - don't show specific batch size values
    ax4.semilogy(actual_batch_sizes, results['grad_errors_custom'], 's-', label='Custom Kernels', linewidth=2, markersize=8)
    ax4.semilogy(actual_batch_sizes, results['grad_errors_cublas'], '^-', label='CUBLAS', linewidth=2, markersize=8)
    ax4.set_xlabel('Actual Batch Size (2*B)')
    ax4.set_ylabel('Max Gradient Error vs Reference')
    ax4.set_title('Gradient Computation Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/performance_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nGenerated plots:")
    print("  - images/execution_times.png - Detailed execution time comparison")
    print("  - images/speedup_comparison.png - Speedup analysis")
    print("  - images/loss_error.png - Loss computation accuracy")
    print("  - images/gradient_error.png - Gradient computation accuracy")
    print("  - images/performance_overview.png - Combined overview")

if __name__ == "__main__":
    run_performance_analysis()
