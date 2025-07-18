#!/bin/bash

# Script per costruire e testare l'implementazione InfoNCE CUDA

echo "=== InfoNCE CUDA Build Script ==="
echo "Implementazione CUDA completa per InfoNCE Loss con batch processing"
echo "Supporta autograd completo di PyTorch e ottimizzazioni GPU"
echo

# Controlli preliminari
echo "1. Verifico l'ambiente..."

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "ERRORE: Python3 non trovato"
    exit 1
fi

# Verifica CUDA
if ! command -v nvcc &> /dev/null; then
    echo "AVVERTIMENTO: nvcc non trovato - potrebbe mancare CUDA Toolkit"
    echo "CUDA Toolkit è necessario per compilare l'estensione"
fi

# Verifica PyTorch
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}')" 2>/dev/null || {
    echo "ERRORE: PyTorch non installato"
    echo "Installa PyTorch con: pip install torch torchvision"
    exit 1
}

# Verifica disponibilità CUDA
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA Devices: {torch.cuda.device_count()}')" 2>/dev/null

echo
echo "2. Pulizia build precedenti..."

# Pulisci build precedenti
rm -rf build/ dist/ *.egg-info/
find . -name "*.so" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Pulizia completata."

echo
echo "3. Build dell'estensione CUDA..."

# Build dell'estensione con architettura CUDA specificata
# Impostiamo TORCH_CUDA_ARCH_LIST per evitare errori di auto-detection
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "Build completata con successo!"
else
    echo "ERRORE: Build fallita"
    echo "Verifica che CUDA Toolkit sia installato correttamente"
    echo "e che la versione PyTorch sia compatibile con CUDA"
    exit 1
fi

echo
echo "4. Test dell'implementazione..."

# Test principale con nuova implementazione
python3 test_implementation.py

if [ $? -eq 0 ]; then
    echo
    echo "=== SUCCESS ==="
    echo "InfoNCE CUDA implementation costruita e testata con successo!"
    echo
    echo "Implementazione disponibile:"
    echo "  from infonce_cuda.infonce_cuda_module import InfoNCELoss, info_nce_loss"
    echo
    echo "Utilizzo per batch completi (raccomandato):"
    echo "  # features: (2*batch_size, feature_dim) normalizzate L2"
    echo "  loss_fn = InfoNCELoss(temperature=0.5)"
    echo "  loss = loss_fn(features)"
    echo
    echo "Oppure funzione diretta:"
    echo "  loss = info_nce_loss(features, temperature=0.5)"
    echo
    echo "Per dettagli completi vedi README.md e report/CUDA_IMPLEMENTATION_REPORT.pdf"
else
    echo "ERRORE: Test falliti"
    echo "Verifica che:"
    echo "  - Le features siano normalizzate L2"
    echo "  - Il batch size sia pari (2*batch_size)"
    echo "  - I tensori siano su GPU"
    exit 1
fi
