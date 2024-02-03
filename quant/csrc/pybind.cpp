#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("gemv_forward_cuda", &gemv_forward_cuda);
  m.def("gemv_forward_cuda_outer_dim", &gemv_forward_cuda_outer_dim);
}