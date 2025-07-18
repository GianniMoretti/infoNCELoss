#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel CUDA per calcolare la matrice di similarità (dot product)
__global__ void similarity_matrix_kernel(const float* features, float* similarity_matrix, 
                                        int batch_size, int feature_dim, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < batch_size && j < batch_size) {
        float dot_product = 0.0f;
        
        // Calcola dot product tra features[i] e features[j]
        for (int d = 0; d < feature_dim; d++) {
            dot_product += features[i * feature_dim + d] * features[j * feature_dim + d];
        }
        
        // Applica temperatura e maschera la diagonale
        if (i == j) {
            similarity_matrix[i * batch_size + j] = -INFINITY;  // Maschera diagonale
        } else {
            similarity_matrix[i * batch_size + j] = dot_product / temperature;
        }
    }
}

// Kernel CUDA per calcolare la cross-entropy loss e il gradiente
__global__ void infonce_forward_backward_kernel(const float* similarity_matrix, const int* labels,
                                                float* loss, float* grad_matrix,
                                                int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        // Calcola softmax per la riga i
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
        
        // Calcola la loss per questa riga
        int positive_idx = labels[i];
        float positive_logit = similarity_matrix[i * batch_size + positive_idx];
        float log_prob = (positive_logit - max_val) - logf(sum_exp);
        
        // Accumula la loss usando atomic add
        atomicAdd(loss, -log_prob / batch_size);
        
        // Calcola il gradiente: P_ij - 1_{j=p(i)}
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

// Kernel CUDA per calcolare il gradiente rispetto alle features
__global__ void features_gradient_kernel(const float* grad_matrix, const float* features,
                                        float* grad_features, int batch_size, int feature_dim,
                                        float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < batch_size && d < feature_dim) {
        float grad_sum = 0.0f;
        
        // Calcola (G + G^T) * Z come nella derivazione matematica
        for (int j = 0; j < batch_size; j++) {
            float g_ij = grad_matrix[i * batch_size + j];
            float g_ji = grad_matrix[j * batch_size + i];
            float z_j = features[j * feature_dim + d];
            
            grad_sum += (g_ij + g_ji) * z_j;
        }
        
        grad_features[i * feature_dim + d] = grad_sum / temperature;
    }
}

// Funzione forward InfoNCE per batch completo
torch::Tensor infonce_cuda_forward(torch::Tensor features, float temperature) {
    // Assicurati che il tensore sia contiguo e su GPU
    features = features.contiguous();
    if (!features.is_cuda()) {
        features = features.cuda();
    }
    
    // Converti a float se necessario
    if (features.dtype() != torch::kFloat) {
        features = features.to(torch::kFloat);
    }
    
    int batch_size = features.size(0);
    int feature_dim = features.size(1);
    
    // Verifica che batch_size sia pari (2*B)
    if (batch_size % 2 != 0) {
        throw std::runtime_error("Batch size must be even (2*B)");
    }
    
    int B = batch_size / 2;
    
    // Crea la matrice di similarità
    auto similarity_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    // Calcola la matrice di similarità
    dim3 block_sim(16, 16);
    dim3 grid_sim((batch_size + block_sim.x - 1) / block_sim.x, (batch_size + block_sim.y - 1) / block_sim.y);
    
    similarity_matrix_kernel<<<grid_sim, block_sim>>>(
        features.data_ptr<float>(),
        similarity_matrix.data_ptr<float>(),
        batch_size,
        feature_dim,
        temperature
    );
    
    // Questa parte è fatta in CPU, in GPU non sarebbe meglio?
    // Crea le labels: per i in [0, B-1] -> positive = i+B, per i in [B, 2B-1] -> positive = i-B
    auto labels = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt).device(features.device()));
    auto labels_ptr = labels.data_ptr<int>();
    
    // Configura le labels su CPU e poi copia su GPU
    std::vector<int> labels_cpu(batch_size);
    for (int i = 0; i < B; i++) {
        labels_cpu[i] = i + B;           // Prima metà punta alla seconda metà
        labels_cpu[i + B] = i;           // Seconda metà punta alla prima metà
    }
    
    cudaMemcpy(labels_ptr, labels_cpu.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Tensore per la loss (questo tensore sara usato per accumulare la loss, però è nella memoria condivisa della GPU percio con atomicAdd potrebbe essere 
    // piu lento? riusciamo ad utilizzare una variabile locale?) )
    auto loss = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    // Tensore temporaneo per il gradiente (non usato nel forward ma necessario per il kernel)
    auto grad_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    // Calcola loss e gradiente
    const int threads_loss = 256;
    const int blocks_loss = (batch_size + threads_loss - 1) / threads_loss;
    
    infonce_forward_backward_kernel<<<blocks_loss, threads_loss>>>(
        similarity_matrix.data_ptr<float>(),
        labels.data_ptr<int>(),
        loss.data_ptr<float>(),
        grad_matrix.data_ptr<float>(),
        batch_size
    );
    
    cudaDeviceSynchronize();
    
    return loss;
}

// Funzione backward InfoNCE per batch completo
torch::Tensor infonce_cuda_backward(torch::Tensor features, float temperature, torch::Tensor grad_output) {
    // Assicurati che i tensori siano contigui e su GPU
    features = features.contiguous();
    grad_output = grad_output.contiguous();
    
    if (!features.is_cuda()) features = features.cuda();
    if (!grad_output.is_cuda()) grad_output = grad_output.cuda();
    
    // Converti a float se necessario
    if (features.dtype() != torch::kFloat) features = features.to(torch::kFloat);
    if (grad_output.dtype() != torch::kFloat) grad_output = grad_output.to(torch::kFloat);
    
    int batch_size = features.size(0);
    int feature_dim = features.size(1);
    
    // Verifica che batch_size sia pari
    if (batch_size % 2 != 0) {
        throw std::runtime_error("Batch size must be even (2*B)");
    }
    
    int B = batch_size / 2;
    
    // Calcola la matrice di similarità
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
    
    // Crea le labels
    auto labels = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt).device(features.device()));
    auto labels_ptr = labels.data_ptr<int>();
    
    std::vector<int> labels_cpu(batch_size);
    for (int i = 0; i < B; i++) {
        labels_cpu[i] = i + B;
        labels_cpu[i + B] = i;
    }
    
    cudaMemcpy(labels_ptr, labels_cpu.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calcola la matrice di gradiente G
    auto grad_matrix = torch::empty({batch_size, batch_size}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    auto dummy_loss = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat).device(features.device()));
    
    const int threads_loss = 256;
    const int blocks_loss = (batch_size + threads_loss - 1) / threads_loss;
    
    // Cosi lo usi da tutte e due le parti ma è brutto e dispendioso
    infonce_forward_backward_kernel<<<blocks_loss, threads_loss>>>(
        similarity_matrix.data_ptr<float>(),
        labels.data_ptr<int>(),
        dummy_loss.data_ptr<float>(),
        grad_matrix.data_ptr<float>(),
        batch_size
    );
    
    // Calcola il gradiente rispetto alle features
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
    
    // Moltiplica per grad_output
    grad_features = grad_features * grad_output.item<float>();
    
    cudaDeviceSynchronize();
    
    return grad_features;
}