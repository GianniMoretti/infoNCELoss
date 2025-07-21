# CuBlaze: InfoNCE Loss CUDA Implementation

**CuBlaze** is a CUDA implementation of **InfoNCE Loss** (Information Noise-Contrastive Estimation) for self-supervised contrastive learning. This implementation offers complete batch processing with native PyTorch autograd integration and support for multiple optimized implementations.

## What is InfoNCE Loss?

InfoNCE Loss is a fundamental loss function in contrastive learning that maximizes agreement between positive pairs of examples and minimizes agreement with negative examples. The mathematical formula is:

```
InfoNCE = -1/N Σᵢ log(exp(sim(zᵢ, z_p(i))/τ) / Σⱼ exp(sim(zᵢ, zⱼ)/τ))
```

Where:
- `N = 2*batch_size` is the total number of examples 
- `zᵢ, zⱼ` are L2 normalized embeddings
- `p(i) = (i + batch_size) % N` identifies the positive pair
- `sim(a,b) = a·b` is cosine similarity (dot product for normalized vectors)
- `τ` is the temperature parameter

## Implementation Features

✅ **Dual Implementation**: Supports both custom CUDA kernels and optimized CUBLAS operations  
✅ **Complete Batch Processing**: Processes feature matrices (2*batch_size, feature_dim)  
✅ **Native Autograd**: Complete integration with PyTorch backward pass  
✅ **Numerical Stability**: Numerically stable computations even with low temperatures  
✅ **GPU Optimized**: Custom CUDA kernels
✅ **SimCLR Compatibility**: Specific design for contrastive learning frameworks  
✅ **Flexible Backend**: Choose between custom CUDA implementation or CUBLAS  

## Project Structure

```
CuBlaze/
├── cublaze/
│   ├── __init__.py                   # Main exports
│   ├── infonce.py                    # Python/autograd implementation
│   └── cuda/
│       ├── infonce_cuda.cu           # Optimized CUDA kernels
│       └── infonce_cuda_wrapp.cpp    # PyBind11 wrapper
├── setup.py                         # Build configuration
├── pyproject.toml                   # Modern packaging configuration
├── build_and_test.sh               # Automatic build+test script
├── test_implementation.py          # Correctness and performance tests
├── performance.py                  # Advanced benchmark analysis
├── reports/                        # Complete technical documentation
├── documentation/                  # LaTeX documentation
├── images/                        # Graphs and visualizations
└── README.md                      # This documentation
```

## Installation

### Prerequisites
- **Python** >= 3.8
- **PyTorch** >= 1.0.0 with CUDA support
- **CUDA Toolkit** >= 11.0
- **Compatible C++ compiler** (g++ >= 7)

### Automatic Build

```bash
# Clone the repository
git clone <repository_url>
cd CuBlaze/

# Run automatic build and tests
chmod +x build_and_test.sh
./build_and_test.sh
```

### Manual Build

```bash
# Clean previous builds
rm -rf build/ *.so

# Build CUDA extension using setuptools
python setup.py build_ext --inplace

# Alternative: modern pip build
pip install -e .

# Test functionality
python test_implementation.py
```

### Package Installation

```bash
# Install as package (recommended for production)
pip install .

# Install in development mode
pip install -e .
```

## Usage

### Basic Implementation

```python
import torch
import torch.nn.functional as F
from cublaze import InfoNCELoss, InfoNCEFunction

# Prepare features for contrastive learning
batch_size = 64
feature_dim = 256

# Simulate encoder output (e.g. from ResNet)
raw_features = torch.randn(2 * batch_size, feature_dim).cuda()

# IMPORTANT: L2 normalize the features
features = F.normalize(raw_features, dim=1)

# Method 1: Using CUDA custom kernels (default, fastest)
loss_fn = InfoNCELoss(temperature=0.5, use_cublas=False)
loss = loss_fn(features)

# Method 2: Using CUBLAS optimized operations  
loss_fn_cublas = InfoNCELoss(temperature=0.5, use_cublas=True)
loss_cublas = loss_fn_cublas(features)

# Method 3: Direct function call (for advanced users)
loss_direct = InfoNCEFunction.apply(features, 0.5, False)  # (features, temperature, use_cublas)

# Calculate gradients
loss.backward()
```

### Backend Selection Guide

**Custom CUDA Kernels** (`use_cublas=False`, default):
- ✅ Faster for large batches
- ✅ Complete control over optimizations

**CUBLAS Operations** (`use_cublas=True`):
- ✅ More stable across different hardware
- ✅ Leverages highly optimized BLAS

### Comparison with Vanilla PyTorch

```python
def pytorch_infonce_reference(features, temperature=0.5):
    """PyTorch reference implementation"""
    device = features.device
    batch_size = features.shape[0] // 2

    # Similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # Mask diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # Labels for positive pairs
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels])
    
    # Cross-entropy with temperature
    similarity_matrix /= temperature
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# Correctness test
features = F.normalize(torch.randn(128, 256), dim=1).cuda()

# CuBlaze implementations
loss_cuda = InfoNCELoss(temperature=0.5, use_cublas=False)(features)
loss_cublas = InfoNCELoss(temperature=0.5, use_cublas=True)(features)

# Reference implementation  
loss_pytorch = pytorch_infonce_reference(features, temperature=0.5)

print(f"CUDA Custom Loss: {loss_cuda.item():.6f}")
print(f"CUBLAS Loss: {loss_cublas.item():.6f}")
print(f"PyTorch Loss: {loss_pytorch.item():.6f}")
print(f"CUDA vs PyTorch: {abs(loss_cuda.item() - loss_pytorch.item()):.8f}")
print(f"CUBLAS vs PyTorch: {abs(loss_cublas.item() - loss_pytorch.item()):.8f}")
```

## API Reference

### InfoNCELoss

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5, use_cublas=False):
        """
        Args:
            temperature (float): Temperature parameter for similarity scaling
            use_cublas (bool): If True, uses CUBLAS operations. If False, uses custom CUDA kernels
        """
        
    def forward(self, features):
        """
        Args:
            features (Tensor): Shape (2*batch_size, feature_dim)
                              MUST be L2 normalized
        Returns:
            Tensor: Scalar loss with gradients
        """
```

### InfoNCEFunction (Advanced)

```python
class InfoNCEFunction(Function):
    @staticmethod
    def apply(features, temperature, use_cublas):
        """
        Direct autograd function call for advanced users
        
        Args:
            features (Tensor): Shape (2*batch_size, feature_dim), L2 normalized
            temperature (float): Temperature scaling parameter
            use_cublas (bool): Backend selection
            
        Returns:
            Tensor: Scalar loss value with autograd support
        """
```

### Package Exports

```python
from cublaze import InfoNCELoss, InfoNCEFunction

# Main classes available at package level
loss_fn = InfoNCELoss(temperature=0.5, use_cublas=False)
```

## Testing and Verification

### Automatic Testing

```bash
# Run complete test suite
python test_implementation.py

# Run performance analysis with visualizations
python performance.py
```

### Test Coverage

The test script verifies:
- ✅ **Numerical correctness**: Comparison with PyTorch reference implementation
- ✅ **Both backends**: Tests both custom CUDA and CUBLAS
- ✅ **Gradient accuracy**: Verifies accurate backward pass
- ✅ **Performance benchmarks**: Execution times for both backends
- ✅ **Error tolerance**: Verifies numerical precision

### Performance Analysis

```bash
# Generate comprehensive performance reports with plots
python performance.py
```

Generates:
- 📊 Speed comparison charts
- 📈 Batch size scalability analysis
- 🔍 Numerical accuracy analysis

### Generated Reports

After running `performance.py`, you'll find in `/images/`:
- `execution_times.png`: Execution time comparison
- `speedup_comparison.png`: Speedup analysis for backends
- `gradient_error.png`: Gradient accuracy
- `loss_error.png`: Loss errors
- `performance_overview.png`: Complete performance overview

## References

- **InfoNCE**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

## License

This project is released under MIT License. See `LICENSE` for details.

---

**Author**: Gianni Moretti  
**Package**: CuBlaze  
**Version**: 0.1  
**Last Updated**: July 2025
