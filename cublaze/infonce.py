import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cublaze.infonce_cuda as infonce_cuda

class InfoNCEFunction(Function):
    """
    Funzione autograd personalizzata per InfoNCE Loss con CUDA
    Implementa la InfoNCE loss come descritto nel paper e nel file LaTeX,
    processando un batch completo di features di forma (2*batch_size, feature_dim)
    """
    @staticmethod
    def forward(ctx, features, temperature):
        # Salva i tensori e parametri per il backward
        ctx.save_for_backward(features)
        ctx.temperature = temperature
        
        # Calcola InfoNCE loss usando CUDA
        loss = infonce_cuda.infonce_forward(features, temperature)
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        # Recupera i tensori e parametri salvati
        features, = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Calcola i gradienti usando CUDA
        grad_features = infonce_cuda.infonce_backward(features, temperature, grad_output)
        
        # Restituisci i gradienti (None per parametri che non richiedono gradienti)
        return grad_features, None

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss implementata in CUDA seguendo la matematica del paper.
    
    Questa implementazione replica esattamente il comportamento del codice Python fornito:
    - Prende in input una matrice di features di dimensione (2*batch_size, feature_dim)
    - Assume che le features siano già normalizzate L2
    - Calcola la matrice di similarità tramite dot product
    - Maschera la diagonale con -inf
    - Applica cross-entropy con le labels appropriate
    - Ogni sample i ha come positivo il sample (i + batch_size) % (2*batch_size)
    
    IMPORTANTE: Le features devono essere già normalizzate L2 prima di essere passate
    """
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Calcola InfoNCE Loss utilizzando CUDA con supporto per autograd
        
        Args:
            features: Tensor di shape (2*batch_size, feature_dim) dove ogni sample i 
                    e sample i+batch_size formano una coppia positiva.
                    DEVE essere già normalizzato L2.
        
        Returns:
            InfoNCE Loss come scalare con gradienti
        """
        # Verifica che il batch size sia pari
        if features.size(0) % 2 != 0:
            raise ValueError("Features tensor must have even batch size (2*batch_size)")
        
        # Assicurati che il tensore sia su GPU
        if not features.is_cuda:
            features = features.cuda()

        # Assicurati che il tensore sia float
        features = features.float()
        
        # Usa la funzione autograd personalizzata
        return InfoNCEFunction.apply(features, self.temperature)

def info_nce_loss(features, temperature=0.5):
    """
    Funzione helper che replica esattamente il codice Python fornito
    
    Args:
        features (Tensor): shape (2*batch_size, feature_dim), where
                           each sample i and i+batch_size form a positive pair.
                           DEVE essere già normalizzato L2.
        temperature (float): temperature scaling parameter.

    Returns:
        torch.Tensor: scalar loss value.
    """
    loss_fn = InfoNCELoss(temperature=temperature)
    return loss_fn(features)

# Funzione helper per confronti diretti (senza autograd)  
def infonce_loss_cuda_no_grad(features, temperature=0.5):
    """
    Calcola InfoNCE Loss senza supporto per autograd (per test/confronti)
    """
    # Implementazione diretta senza autograd
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
