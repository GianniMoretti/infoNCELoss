#!/usr/bin/env python3
"""
Test script per la nuova implementazione InfoNCE CUDA
Replica esattamente il codice Python fornito e confronta i risultati
"""

import torch
import torch.nn.functional as F
from cublaze.infonce import InfoNCELoss, info_nce_loss

def info_nce_loss_reference(features, temperature=0.5):
    """
    Implementazione di riferimento in PyTorch puro (dal codice fornito dall'utente)
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
    Test di correttezza confrontando con l'implementazione di riferimento
    """
    print("=== Test di Correttezza ===")
    
    # Parametri del test
    batch_size = 4  # Quindi 2*batch_size = 8
    feature_dim = 128
    temperature = 0.5
    
    # Genera features casuali
    torch.manual_seed(42)
    features = torch.randn(2 * batch_size, feature_dim, device='cuda', requires_grad=True)
    features = F.normalize(features, dim=1)  # Normalizza L2
    
    # Test dell'implementazione CUDA
    print("Testing CUDA implementation...")
    features_cuda = features.clone().detach().requires_grad_(True)
    loss_cuda = info_nce_loss(features_cuda, temperature)
    
    # Test dell'implementazione di riferimento
    print("Testing reference implementation...")
    features_ref = features.clone().detach().requires_grad_(True)
    loss_ref = info_nce_loss_reference(features_ref, temperature)
    
    print(f"CUDA Loss: {loss_cuda.item():.6f}")
    print(f"Reference Loss: {loss_ref.item():.6f}")
    print(f"Differenza: {abs(loss_cuda.item() - loss_ref.item()):.8f}")
    
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
    Test di performance confrontando i tempi di esecuzione
    """
    print("\n=== Test di Performance ===")
    
    batch_size = 64  # 2*batch_size = 128
    feature_dim = 512
    temperature = 0.5
    num_iterations = 100
    
    # Genera features casuali
    features = torch.randn(2 * batch_size, feature_dim, device='cuda')
    
    # Warm-up
    for _ in range(10):
        _ = info_nce_loss(features, temperature)
        _ = info_nce_loss_reference(features, temperature)
    
    torch.cuda.synchronize()
    
    # Test CUDA implementation
    import time
    start_time = time.time()
    for _ in range(num_iterations):
        loss_cuda = info_nce_loss(features, temperature)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # Test reference implementation
    start_time = time.time()
    for _ in range(num_iterations):
        loss_ref = info_nce_loss_reference(features, temperature)
    torch.cuda.synchronize()
    ref_time = time.time() - start_time
    
    print(f"CUDA time: {cuda_time:.4f}s ({cuda_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Reference time: {ref_time:.4f}s ({ref_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Speedup: {ref_time/cuda_time:.2f}x")

def test_different_sizes():
    """
    Test con diverse dimensioni del batch
    """
    print("\n=== Test con Diverse Dimensioni ===")
    
    temperature = 0.5
    feature_dim = 256
    
    batch_sizes = [2, 8, 16, 32]  # Questi sono i B, quindi 2*B sarà il batch effettivo
    
    for B in batch_sizes:
        print(f"\nTesting batch_size = {B} (total samples = {2*B})")
        
        features = torch.randn(2 * B, feature_dim, device='cuda', requires_grad=True)
        
        try:
            loss_cuda = info_nce_loss(features, temperature)
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
