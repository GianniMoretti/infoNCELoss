import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cublaze.infonce_cuda as infonce_cuda

class InfoNCEFunction(Function):
    """
    Custom autograd function for InfoNCE Loss with CUDA
    Implements InfoNCE loss as described in the paper and LaTeX file,
    processing a complete batch of features with shape (2*batch_size, feature_dim)
    """
    @staticmethod
    def forward(ctx, features, temperature):
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(features)
        ctx.temperature = temperature
        
        # Calculate InfoNCE loss using CUDA
        loss = infonce_cuda.infonce_forward(features, temperature)
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and parameters
        features, = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Calculate gradients using CUDA
        grad_features = infonce_cuda.infonce_backward(features, temperature, grad_output)
        
        # Return gradients (None for parameters that don't require gradients)
        return grad_features, None

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss implemented in CUDA following the paper mathematics.
    
    This implementation exactly replicates the behavior of the provided Python code:
    - Takes input feature matrix of dimension (2*batch_size, feature_dim)
    - Assumes features are already L2 normalized
    - Calculates similarity matrix through dot product
    - Masks the diagonal with -inf
    - Applies cross-entropy with appropriate labels
    - Each sample i has as positive the sample (i + batch_size) % (2*batch_size)
    
    IMPORTANT: Features must be L2 normalized before being passed
    """
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Calculates InfoNCE Loss using CUDA with autograd support
        
        Args:
            features: Tensor of shape (2*batch_size, feature_dim) where each sample i 
                    and sample i+batch_size form a positive pair.
                    MUST be already L2 normalized.
        
        Returns:
            InfoNCE Loss as scalar with gradients
        """
        # Verify that batch size is even
        if features.size(0) % 2 != 0:
            raise ValueError("Features tensor must have even batch size (2*batch_size)")
        
        # Ensure tensor is on GPU
        if not features.is_cuda:
            features = features.cuda()

        # Ensure tensor is float
        features = features.float()
        
        # Use custom autograd function
        return InfoNCEFunction.apply(features, self.temperature)
