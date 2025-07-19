#include <torch/extension.h>

// Dichiara le funzioni definite nel .cu - updated signatures with max_vals and sum_exps optimization
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> infonce_cuda_forward(torch::Tensor features, float temperature, bool use_cublas);
torch::Tensor infonce_cuda_backward(torch::Tensor features, torch::Tensor similarity_matrix, 
                                    torch::Tensor labels, torch::Tensor max_vals, torch::Tensor sum_exps,
                                    float temperature, torch::Tensor grad_output, bool use_cublas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("infonce_forward", &infonce_cuda_forward, "InfoNCE Loss forward (CUDA) - returns (loss, similarity_matrix, labels, max_vals, sum_exps)");
    m.def("infonce_backward", &infonce_cuda_backward, "InfoNCE Loss backward (CUDA) - optimized with pre-computed max_vals and sum_exps");
}
