#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
at::native::native_add_dropout_fuse_cuda(
    const torch::Tensor& input1,
    const torch::Tensor& residual1,
    const torch::Tensor& input2,
    const torch::Tensor& residual2,
    double p,
    c10::optional<bool> train);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
at::native::native_add_dropout_fuse_2_cuda(
    const torch::Tensor& grad_output1,
    const torch::Tensor& mask1,
    const torch::Tensor& grad_output2,
    const torch::Tensor& mask2,
    double scale);

// C++ interface

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
add_dropout_fuse_forward(
    torch::Tensor& input1,
    torch::Tensor& residual1,
    torch::Tensor& input2,
    torch::Tensor& residual2,
    double prob,
    c10::optional<bool> training) {
  CHECK_INPUT(input1);
  CHECK_INPUT(residual1);
  CHECK_INPUT(input2);
  CHECK_INPUT(residual2);

  // print tensor debug string
  printf("add_dropout_fuse_forward\n");
  printf("input1: ");
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 4; k++) {
        printf("%f ", input1[i][j][k].item<float>());
      }
    }
  }
  printf("\n");
  printf("residual1: ");
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 4; k++) {
        printf("%f ", residual1[i][j][k].item<float>());
      }
    }
  }
  printf("\n");

  return at::native::native_add_dropout_fuse_cuda(
      input1, residual1, input2, residual2, prob, training);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
add_dropout_fuse_backward(
    torch::Tensor& grad_output1,
    torch::Tensor& mask1,
    torch::Tensor& grad_output2,
    torch::Tensor& mask2,
    double scale) {
  CHECK_INPUT(grad_output1);
  CHECK_INPUT(mask1);
  CHECK_INPUT(grad_output2);
  CHECK_INPUT(mask2);

  return at::native::native_add_dropout_fuse_2_cuda(
      grad_output1, mask1, grad_output2, mask2, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward", &add_dropout_fuse_forward, "Add dropout fuse forward (CUDA)");
  m.def(
      "backward",
      &add_dropout_fuse_backward,
      "Add dropout fuse backward (CUDA)");
}