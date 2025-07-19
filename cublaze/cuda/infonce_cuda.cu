#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to compute similarity matrix (dot product)
__global__ void similarity_matrix_kernel(const float* features, float* similarity_matrix, 
                                        int batch_size, int feature_dim, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < batch_size && j < batch_size) {
        float dot_product = 0.0f;
        
        // Calculate dot product between features[i] and features[j]
        for (int d = 0; d < feature_dim; d++) {
            dot_product += features[i * feature_dim + d] * features[j * feature_dim + d];
        }
        
        // Apply temperature and mask diagonal
        if (i == j) {
            similarity_matrix[i * batch_size + j] = -INFINITY;  // Mask diagonal
        } else {
            similarity_matrix[i * batch_size + j] = dot_product / temperature;
        }
    }
}

// CUDA kernel to compute InfoNCE loss (forward pass only)
__global__ void infonce_forward_kernel(const float* similarity_matrix, const int* labels, float* loss, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        // Calculate softmax for row i
        float max_val = -INFINITY;
        for (int j = 0; j < batch_size; j++) {
            float val = similarity_matrix[i * batch_size + j];
            if (val > max_val && val != -INFINITY) {
                max_val = val;
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float val = similarity_matrix[i * batch_size + j];
            if (val != -INFINITY) {
                sum_exp += expf(val - max_val);
            }
        }
        
        // Calculate loss for this row
        int positive_idx = labels[i];
        float positive_logit = similarity_matrix[i * batch_size + positive_idx];
        float log_prob = (positive_logit - max_val) - logf(sum_exp);
        
        // Accumulate loss using atomic add
        atomicAdd(loss, -log_prob / batch_size);
    }
}

// CUDA kernel to compute gradients for InfoNCE (backward pass only)
__global__ void infonce_backward_kernel(const float* similarity_matrix, const int* labels,
                                       float* grad_matrix, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        // Calculate softmax for row i
        float max_val = -INFINITY;
        for (int j = 0; j < batch_size; j++) {
            float val = similarity_matrix[i * batch_size + j];
            if (val > max_val && val != -INFINITY) {
                max_val = val;
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float val = similarity_matrix[i * batch_size + j];
            if (val != -INFINITY) {
                sum_exp += expf(val - max_val);
            }
        }
        
        // Calculate gradient: P_ij - 1_{j=p(i)}
        int positive_idx = labels[i];
        for (int j = 0; j < batch_size; j++) {
            float val = similarity_matrix[i * batch_size + j];
            if (val != -INFINITY) {
                float prob = expf(val - max_val) / sum_exp;
                float grad_val = prob - (j == positive_idx ? 1.0f : 0.0f);
                grad_matrix[i * batch_size + j] = grad_val / batch_size;
            } else {
                grad_matrix[i * batch_size + j] = 0.0f;
            }
        }
    }
}

// CUDA kernel to compute gradient with respect to features
__global__ void features_gradient_kernel(const float* grad_matrix, const float* features,
                                        float* grad_features, int batch_size, int feature_dim,
                                        float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < batch_size && d < feature_dim) {
        float grad_sum = 0.0f;
        
        // Calculate (G + G^T) * Z as in the mathematical derivation
        for (int j = 0; j < batch_size; j++) {
            float g_ij = grad_matrix[i * batch_size + j];
            float g_ji = grad_matrix[j * batch_size + i];
            float z_j = features[j * feature_dim + d];
            
            grad_sum += (g_ij + g_ji) * z_j;
        }
        
        grad_features[i * feature_dim + d] = grad_sum / temperature;
    }
}

// InfoNCE forward function for complete batch
torch::Tensor infonce_cuda_forward(torch::Tensor features, float temperature) {
    // Ensure tensor is contiguous and on GPU
    features = features.contiguous();
    if (!features.is_cuda()) {
        features = features.cuda();
    }
    
    // Convert to float if necessary
    if (features.dtype() != torch::kFloat) {
        features = features.to(torch::kFloat);
    }
    
    int batch_size = features.size(0);
    int feature_dim = features.size(1);
    
    // Verify that batch_size is even (2*B)
    if (batch_size % 2 != 0) {
        throw std::runtime_error("Batch size must be even (2*B)");
    }
    
    int B = batch_size / 2;
    
    // Create similarity matrix
    auto similarity_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    // Calculate similarity matrix
    dim3 block_sim(16, 16);
    dim3 grid_sim((batch_size + block_sim.x - 1) / block_sim.x, (batch_size + block_sim.y - 1) / block_sim.y);
    
    similarity_matrix_kernel<<<grid_sim, block_sim>>>(
        features.data_ptr<float>(),
        similarity_matrix.data_ptr<float>(),
        batch_size,
        feature_dim,
        temperature
    );
    
    // TODO: This part is done on CPU, wouldn't it be better on GPU?
    // Create labels: for i in [0, B-1] -> positive = i+B, for i in [B, 2B-1] -> positive = i-B
    auto labels = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt).device(features.device()));
    auto labels_ptr = labels.data_ptr<int>();
    
    // Configure labels on CPU then copy to GPU
    std::vector<int> labels_cpu(batch_size);
    for (int i = 0; i < B; i++) {
        labels_cpu[i] = i + B;           // First half points to second half
        labels_cpu[i + B] = i;           // Second half points to first half
    }
    
    cudaMemcpy(labels_ptr, labels_cpu.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Tensor for loss (this tensor will be used to accumulate loss, but it's in GPU shared memory so atomicAdd might be
    // slower? Can we use a local variable instead?)
    auto loss = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    // Calculate loss
    const int threads_loss = 256;
    const int blocks_loss = (batch_size + threads_loss - 1) / threads_loss;
    
    infonce_forward_kernel<<<blocks_loss, threads_loss>>>(
        similarity_matrix.data_ptr<float>(),
        labels.data_ptr<int>(),
        loss.data_ptr<float>(),
        batch_size
    );
    
    cudaDeviceSynchronize();
    
    return loss;
}

// InfoNCE backward function for complete batch
torch::Tensor infonce_cuda_backward(torch::Tensor features, float temperature, torch::Tensor grad_output) {
    // Ensure tensors are contiguous and on GPU
    features = features.contiguous();
    grad_output = grad_output.contiguous();
    
    if (!features.is_cuda()) features = features.cuda();
    if (!grad_output.is_cuda()) grad_output = grad_output.cuda();
    
    // Convert to float if necessary
    if (features.dtype() != torch::kFloat) features = features.to(torch::kFloat);
    if (grad_output.dtype() != torch::kFloat) grad_output = grad_output.to(torch::kFloat);
    
    int batch_size = features.size(0);
    int feature_dim = features.size(1);
    
    // Verify that batch_size is even
    if (batch_size % 2 != 0) {
        throw std::runtime_error("Batch size must be even (2*B)");
    }
    
    int B = batch_size / 2;
    
    // Calculate similarity matrix
    auto similarity_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    dim3 block_sim(16, 16);
    dim3 grid_sim((batch_size + block_sim.x - 1) / block_sim.x, (batch_size + block_sim.y - 1) / block_sim.y);
    
    similarity_matrix_kernel<<<grid_sim, block_sim>>>(
        features.data_ptr<float>(),
        similarity_matrix.data_ptr<float>(),
        batch_size,
        feature_dim,
        temperature
    );
    
    // Create labels
    auto labels = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt).device(features.device()));
    auto labels_ptr = labels.data_ptr<int>();
    
    std::vector<int> labels_cpu(batch_size);
    for (int i = 0; i < B; i++) {
        labels_cpu[i] = i + B;
        labels_cpu[i + B] = i;
    }
    
    cudaMemcpy(labels_ptr, labels_cpu.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate gradient matrix G
    auto grad_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    const int threads_loss = 256;
    const int blocks_loss = (batch_size + threads_loss - 1) / threads_loss;
    
    infonce_backward_kernel<<<blocks_loss, threads_loss>>>(
        similarity_matrix.data_ptr<float>(),
        labels.data_ptr<int>(),
        grad_matrix.data_ptr<float>(),
        batch_size
    );
    
    // Calculate gradient with respect to features
    auto grad_features = torch::zeros_like(features);
    
    dim3 block_grad(16, 16);
    dim3 grid_grad((batch_size + block_grad.x - 1) / block_grad.x, (feature_dim + block_grad.y - 1) / block_grad.y);
    
    features_gradient_kernel<<<grid_grad, block_grad>>>(
        grad_matrix.data_ptr<float>(),
        features.data_ptr<float>(),
        grad_features.data_ptr<float>(),
        batch_size,
        feature_dim,
        temperature
    );
    
    // Multiply by grad_output
    grad_features = grad_features * grad_output.item<float>();
    
    cudaDeviceSynchronize();
    
    return grad_features;
}