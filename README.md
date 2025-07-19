# CuBlaze: InfoNCE Loss CUDA Implementation

**CuBlaze** è un'implementazione CUDA di **InfoNCE Loss** (Information Noise-Contrastive Estimation) per self-supervised contrastive learning. Questa implementazione offre elaborazione batch completa con integrazione nativa di PyTorch autograd e supporto per multiple implementazioni ottimizzate.

## What is InfoNCE Loss?

InfoNCE Loss è una funzione di perdita fondamentale nel contrastive learning che massimizza l'accordo tra coppie positive di esempi e minimizza l'accordo con esempi negativi. La formula matematica è:

```
InfoNCE = -1/N Σᵢ log(exp(sim(zᵢ, z_p(i))/τ) / Σⱼ exp(sim(zᵢ, zⱼ)/τ))
```

Dove:
- `N = 2*batch_size` è il numero totale di esempi 
- `zᵢ, zⱼ` sono embedding normalizzati L2
- `p(i) = (i + batch_size) % N` identifica la coppia positiva
- `sim(a,b) = a·b` è la similarità coseno (dot product per vettori normalizzati)
- `τ` è il parametro temperatura

## Implementation Features

✅ **Dual Implementation**: Supporta sia kernels CUDA custom che operazioni CUBLAS ottimizzate  
✅ **Complete Batch Processing**: Elabora matrici di feature (2*batch_size, feature_dim)  
✅ **Native Autograd**: Integrazione completa con backward pass di PyTorch  
✅ **Numerical Stability**: Calcoli numericamente stabili anche con temperature basse  
✅ **GPU Optimized**: Kernels CUDA personalizzati
✅ **SimCLR Compatibility**: Design specifico per framework di contrastive learning  
✅ **Flexible Backend**: Scegli tra implementazione custom CUDA o CUBLAS  

## Project Structure

```
CuBlaze/
├── cublaze/
│   ├── __init__.py                   # Export principali
│   ├── infonce.py                    # Implementazione Python/autograd
│   └── cuda/
│       ├── infonce_cuda.cu           # Kernels CUDA ottimizzati
│       └── infonce_cuda_wrapp.cpp    # Wrapper PyBind11
├── setup.py                         # Configurazione build
├── pyproject.toml                   # Configurazione packaging moderno
├── build_and_test.sh               # Script automatico build+test
├── test_implementation.py          # Test correttezza e performance
├── performance.py                  # Analisi benchmark avanzata
├── reports/                        # Documentazione tecnica completa
├── documentation/                  # Documentazione LaTeX
├── images/                        # Grafici e visualizzazioni
└── README.md                      # Questa documentazione
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
- ✅ Più veloce per batch grandi
- ✅ Controllo completo su ottimizzazioni

**CUBLAS Operations** (`use_cublas=True`):
- ✅ Più stabile su hardware diverso
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

Il script di test verifica:
- ✅ **Numerical correctness**: Confronto con implementazione PyTorch di riferimento
- ✅ **Both backends**: Test sia custom CUDA che CUBLAS
- ✅ **Gradient accuracy**: Verifica backward pass accurato
- ✅ **Performance benchmarks**: Tempi di esecuzione per entrambi i backend
- ✅ **Error tolerance**: Verifica precisione numerica

### Performance Analysis

```bash
# Generate comprehensive performance reports with plots
python performance.py
```

Genera:
- 📊 Grafici di confronto velocità
- 📈 Analisi scalabilità batch size
- 🔍 Analisi accuratezza numerica

### Generated Reports

Dopo aver eseguito `performance.py`, troverai in `/images/`:
- `execution_times.png`: Confronto tempi esecuzione
- `speedup_comparison.png`: Analisi speedup per backend
- `gradient_error.png`: Accuratezza gradienti
- `loss_error.png`: Errori di loss
- `performance_overview.png`: Overview completo prestazioni

## References

- **InfoNCE**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

## License

This project is released under MIT License. See `LICENSE` for details.

---

**Author**: Gianni Moretti  
**Package**: CuBlaze  
**Version**: 0.1  
**Last Updated**: July 2025
