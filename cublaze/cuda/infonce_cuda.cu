#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// =============================================================================
// DYNAMIC BLOCK SIZE CONFIGURATION HELPERS
// =============================================================================

// Helper function to calculate optimal block size for 1D kernels
__host__ int calculate_optimal_block_size_1d(int total_elements) {
    
    // For small problems, use smaller blocks
    if (total_elements <= 32) return 32;
    else if (total_elements <= 64) return 64;
    else if (total_elements <= 128) return 128;
    else if (total_elements <= 1024) return 256;
    else if (total_elements <= 4096) return 512;
    
    // For very large problems, use maximum block size
    return 1024;
}

// Helper function to calculate optimal block configuration for 2D kernels
__host__ dim3 calculate_optimal_block_size_2d(int dim1, int dim2) {
    // For 2D kernels, balance between the two dimensions
    
    if (dim1 <= 16 && dim2 <= 16) {
        return dim3(16, 16);  // 256 threads total
    }
    
    if (dim1 <= 32 && dim2 <= 32) {
        return dim3(min(dim1, 32), min(dim2, 32));
    }
    
    // For larger dimensions, use standard configurations
    if (dim1 >= dim2) {
        if (dim2 <= 16) return dim3(32, 8);   // 256 threads
        else return dim3(16, 16);             // 256 threads
    } else {
        if (dim1 <= 16) return dim3(8, 32);   // 256 threads  
        else return dim3(16, 16);             // 256 threads
    }
}

__host__ int calculate_grid_size_1d(int total_elements, int block_size) {
    return (total_elements + block_size - 1) / block_size;
}

__host__ dim3 calculate_grid_size_2d(int dim1, int dim2, dim3 block_size) {
    return dim3((dim1 + block_size.x - 1) / block_size.x, 
                (dim2 + block_size.y - 1) / block_size.y);
}

// =============================================================================
// KERNEL FUNCTIONS
// =============================================================================

__global__ void create_labels_kernel(int* labels, int batch_size, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        // First half points to second half, second half points to first half
        labels[i] = (i < B) ? (i + B) : (i - B);
    }
}

// CUDA kernel to compute similarity matrix (dot product)
__global__ void similarity_matrix_kernel(const float* features, float* similarity_matrix, 
                                        int batch_size, int feature_dim, float temperature) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * batch_size;
    
    if (tid < total_elements) {
        int i = tid / batch_size;
        int j = tid % batch_size;
        
        float dot_product = 0.0f;
        
        // Calculate dot product between features[i] and features[j]
        for (int d = 0; d < feature_dim; d++) {
            dot_product += features[i * feature_dim + d] * features[j * feature_dim + d];
        }
        
        // Apply temperature and mask diagonal
        if (i == j) {
            similarity_matrix[tid] = -INFINITY;  // Mask diagonal
        } else {
            similarity_matrix[tid] = dot_product / temperature;
        }
    }
}

__global__ void mask_diagonal_kernel(float* similarity_matrix, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        // Mask diagonal element to -inf
        similarity_matrix[i * batch_size + i] = -INFINITY;
    }
}

__global__ void infonce_forward_kernel(const float* similarity_matrix, const int* labels, 
                                                  float* loss, float* max_vals, float* sum_exps, 
                                                  int batch_size) {
    // Dynamic shared memory allocation based on block size
    extern __shared__ float shared_loss[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    shared_loss[tid] = 0.0f;
    
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
        
        //Save max_val and sum_exp for backward pass
        max_vals[i] = max_val;
        sum_exps[i] = sum_exp;
        
        // Calculate loss for this row
        int positive_idx = labels[i];
        float positive_logit = similarity_matrix[i * batch_size + positive_idx];
        float log_prob = (positive_logit - max_val) - logf(sum_exp);
        
        // Store local loss in shared memory (normalized by batch size)
        shared_loss[tid] = -log_prob / batch_size;
    }
    
    // Synchronize to ensure all threads have written to shared memory
    __syncthreads();

    if (tid == 0) {
        float block_loss = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            block_loss += shared_loss[i];
        }
        atomicAdd(loss, block_loss);
    } 
}

__global__ void infonce_backward_kernel(const float* similarity_matrix, const int* labels,
                                                    const float* max_vals, const float* sum_exps,
                                                    float* grad_matrix, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * batch_size;
    
    if (tid < total_elements) {
        int i = tid / batch_size;
        int j = tid % batch_size;
        
        // Use pre-computed values from forward pass
        float max_val = max_vals[i];
        float sum_exp = sum_exps[i];
        
        // Calculate gradient: P_ij - 1_{j=p(i)}
        int positive_idx = labels[i];
        float val = similarity_matrix[tid];
        
        if (val != -INFINITY) {
            float prob = expf(val - max_val) / sum_exp;
            float grad_val = prob - (j == positive_idx ? 1.0f : 0.0f);
            grad_matrix[tid] = grad_val / batch_size;
        } else {
            grad_matrix[tid] = 0.0f;
        }
    }
}

//CUDA kernel to compute gradient with respect to features - COALESCED VERSION (NO CACHING)
__global__ void features_gradient_kernel(
    const float* grad_matrix, const float* features,
    float* grad_features, int batch_size, int feature_dim,
    float temperature) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < batch_size && d < feature_dim) {
        float grad_sum = 0.0f;
        
        for (int j = 0; j < batch_size; j++) {
            float g_ij = grad_matrix[i * batch_size + j];
            float g_ji = grad_matrix[j * batch_size + i];
            float z_j = features[j * feature_dim + d];
            
            grad_sum += (g_ij + g_ji) * z_j;
        }
        
        // Coalesced write: threads with consecutive (i,d) write to consecutive memory
        grad_features[i * feature_dim + d] = grad_sum / temperature;
    }
}

// =============================================================================
// BACKWARD AND FORWARD FUNCTIONS
// =============================================================================

// InfoNCE forward function for complete batch - now returns (loss, similarity_matrix, labels, max_vals, sum_exps)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> infonce_cuda_forward(torch::Tensor features, float temperature, bool use_cublas) {
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
    
    torch::Tensor similarity_matrix;
    
    if (use_cublas) {
        // Use optimized CUBLAS implementation
        similarity_matrix = torch::mm(features, features.t()); // features @ features.T
        
        // Apply temperature scaling
        similarity_matrix = similarity_matrix / temperature;
        
        const int threads_mask = calculate_optimal_block_size_1d(batch_size);
        const int blocks_mask = calculate_grid_size_1d(batch_size, threads_mask);
        
        mask_diagonal_kernel<<<blocks_mask, threads_mask>>>(
            similarity_matrix.data_ptr<float>(),
            batch_size
        );
    } else {
        // Use custom kernel implementation
        similarity_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
        
        // Calculate similarity matrix USING CUSTOM KERNEL
        dim3 block_sim(calculate_optimal_block_size_1d(batch_size * batch_size), 1);
        dim3 grid_sim(calculate_grid_size_1d(batch_size * batch_size, block_sim.x), 1);
        
        similarity_matrix_kernel<<<grid_sim, block_sim>>>(
            features.data_ptr<float>(),
            similarity_matrix.data_ptr<float>(),
            batch_size,
            feature_dim,
            temperature
        );
    }
    
    auto labels = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt).device(features.device()));
    
    //Launch kernel to create labels 
    const int threads_labels = calculate_optimal_block_size_1d(batch_size);
    const int blocks_labels = calculate_grid_size_1d(batch_size, threads_labels);
    
    create_labels_kernel<<<blocks_labels, threads_labels>>>(
        labels.data_ptr<int>(),
        batch_size,
        B
    );
    
    auto loss = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    //Create tensors to store max_vals and sum_exps for backward pass
    auto max_vals = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    auto sum_exps = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    const int threads_loss = calculate_optimal_block_size_1d(batch_size);
    const int blocks_loss = calculate_grid_size_1d(batch_size, threads_loss);
    
    // Calculate shared memory size needed (one float per thread)
    size_t shared_memory_size = threads_loss * sizeof(float);
    
    infonce_forward_kernel<<<blocks_loss, threads_loss, shared_memory_size>>>(
        similarity_matrix.data_ptr<float>(),
        labels.data_ptr<int>(),
        loss.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        sum_exps.data_ptr<float>(),
        batch_size
    );
    
    cudaDeviceSynchronize();
    
    // Return loss, similarity_matrix, labels, max_vals, and sum_exps for backward pass optimization
    return std::make_tuple(loss, similarity_matrix, labels, max_vals, sum_exps);
}

// InfoNCE backward function for complete batch
torch::Tensor infonce_cuda_backward(torch::Tensor features, torch::Tensor similarity_matrix, 
                                    torch::Tensor labels, torch::Tensor max_vals, torch::Tensor sum_exps,
                                    float temperature, torch::Tensor grad_output, bool use_cublas) {
    // Ensure tensors are contiguous and on GPU
    features = features.contiguous();
    similarity_matrix = similarity_matrix.contiguous();
    labels = labels.contiguous();
    max_vals = max_vals.contiguous();
    sum_exps = sum_exps.contiguous();
    grad_output = grad_output.contiguous();
    
    if (!features.is_cuda()) features = features.cuda();
    if (!similarity_matrix.is_cuda()) similarity_matrix = similarity_matrix.cuda();
    if (!labels.is_cuda()) labels = labels.cuda();
    if (!max_vals.is_cuda()) max_vals = max_vals.cuda();
    if (!sum_exps.is_cuda()) sum_exps = sum_exps.cuda();
    if (!grad_output.is_cuda()) grad_output = grad_output.cuda();
    
    // Convert to float if necessary
    if (features.dtype() != torch::kFloat) features = features.to(torch::kFloat);
    if (similarity_matrix.dtype() != torch::kFloat) similarity_matrix = similarity_matrix.to(torch::kFloat);
    if (max_vals.dtype() != torch::kFloat) max_vals = max_vals.to(torch::kFloat);
    if (sum_exps.dtype() != torch::kFloat) sum_exps = sum_exps.to(torch::kFloat);
    if (grad_output.dtype() != torch::kFloat) grad_output = grad_output.to(torch::kFloat);
    // Note: labels are int, so no conversion needed
    
    int batch_size = features.size(0);
    int feature_dim = features.size(1);
    
    // Calculate gradient matrix G
    auto grad_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    //Calculate gradient matrix using pre-computed max_vals and sum_exps
    const int threads_backward = calculate_optimal_block_size_1d(batch_size * batch_size);
    const int blocks_backward = calculate_grid_size_1d(batch_size * batch_size, threads_backward);
    
    infonce_backward_kernel<<<blocks_backward, threads_backward>>>(
        similarity_matrix.data_ptr<float>(),
        labels.data_ptr<int>(),
        max_vals.data_ptr<float>(),
        sum_exps.data_ptr<float>(),
        grad_matrix.data_ptr<float>(),
        batch_size
    );
    
    torch::Tensor grad_features;
    
    if (use_cublas) {
        // Use optimized matrix operations
        auto grad_matrix_symmetric = grad_matrix + grad_matrix.t();
        grad_features = torch::mm(grad_matrix_symmetric, features) / temperature;
    } else {
        // Use custom kernel implementation
        grad_features = torch::zeros_like(features);
        
        //Calculate gradient with custom kernel
        dim3 block_grad = calculate_optimal_block_size_2d(batch_size, feature_dim);
        dim3 grid_grad = calculate_grid_size_2d(batch_size, feature_dim, block_grad);
        
        // Calculate shared memory size: features cache + gradients cache
        size_t shared_memory_size = (block_grad.y * feature_dim + block_grad.x * block_grad.y) * sizeof(float);
        
        features_gradient_kernel<<<grid_grad, block_grad, shared_memory_size>>>(
            grad_matrix.data_ptr<float>(),
            features.data_ptr<float>(),
            grad_features.data_ptr<float>(),
            batch_size,
            feature_dim,
            temperature
        );
    }
    
    // Multiply by grad_output
    grad_features = grad_features * grad_output.item<float>();
    
    cudaDeviceSynchronize();
    
    return grad_features;
}