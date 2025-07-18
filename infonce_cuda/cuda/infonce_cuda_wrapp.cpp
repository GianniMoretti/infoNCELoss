#include <torch/extension.h>

// Dichiara le funzioni definite nel .cu
torch::Tensor infonce_cuda_forward(torch::Tensor features, float temperature);
torch::Tensor infonce_cuda_backward(torch::Tensor features, float temperature, torch::Tensor grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("infonce_forward", &infonce_cuda_forward, "InfoNCE Loss forward (CUDA)");
    m.def("infonce_backward", &infonce_cuda_backward, "InfoNCE Loss backward (CUDA)");
}
