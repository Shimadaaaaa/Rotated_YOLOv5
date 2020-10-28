#include <torch/extension.h>
#include <torch/types.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor")

at::Tensor rnms_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor rnms(const at::Tensor& dets, const at::Tensor& scores, const float threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
  return rnms_cuda(b, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("r_nms", &rnms, "r_nms rnms");
}