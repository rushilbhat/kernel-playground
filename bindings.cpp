#include <torch/extension.h>

#ifdef COMPILE_MATMUL_BIAS_RELU
torch::Tensor matmul_bias_relu(
    torch::Tensor a, torch::Tensor b, torch::Tensor bias
);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

#ifdef COMPILE_MATMUL_BIAS_RELU
    m.def("matmul_bias_relu", matmul_bias_relu);
#endif
}