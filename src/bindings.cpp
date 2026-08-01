#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

torch::Tensor grouped_lap_cuda(
    const torch::Tensor& cost,
    const torch::Tensor& descriptors,
    const std::vector<int64_t>& bucket_counts,
    int64_t total_small,
    int64_t total_large);

namespace {

void check_cuda_input(
    const torch::Tensor& cost,
    const torch::Tensor& descriptors,
    const std::vector<int64_t>& bucket_counts,
    int64_t total_small,
    int64_t total_large) {
  TORCH_CHECK(cost.is_cuda(), "cost must be a CUDA tensor");
  TORCH_CHECK(cost.dim() == 3, "cost must have shape [N, M, Y]");
  TORCH_CHECK(cost.layout() == at::kStrided,
              "cost must use a strided layout");
  TORCH_CHECK(
      cost.scalar_type() == torch::kFloat ||
          cost.scalar_type() == torch::kDouble,
      "cost must have dtype torch.float32 or torch.float64");

  TORCH_CHECK(descriptors.device() == cost.device(),
              "descriptors and cost must be on the same CUDA device");
  TORCH_CHECK(descriptors.is_contiguous(), "descriptors must be contiguous");
  TORCH_CHECK(descriptors.scalar_type() == torch::kLong,
              "descriptors must have dtype torch.int64");
  TORCH_CHECK(descriptors.dim() == 2 && descriptors.size(1) == 4,
              "descriptors must have shape [K, 4]");

  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  constexpr int64_t kMaxCooperativeDimension = kIntMax - 255;
  TORCH_CHECK(cost.size(0) <= kMaxCooperativeDimension &&
                  cost.size(2) <= kMaxCooperativeDimension &&
                  cost.size(1) <= kIntMax,
              "N and Y must be at most INT_MAX - 255, and M must fit in a "
              "signed 32-bit integer");
  TORCH_CHECK(bucket_counts.size() == 4,
              "bucket_counts must contain 4 entries");
  int64_t descriptor_count = 0;
  for (const int64_t count : bucket_counts) {
    TORCH_CHECK(count >= 0, "bucket counts must be non-negative");
    TORCH_CHECK(descriptor_count <= kIntMax - count,
                "number of bucket descriptors exceeds INT_MAX");
    descriptor_count += count;
    TORCH_CHECK(
        count == 0 || cost.size(1) <= kIntMax / count,
        "number of (group, M) tasks in a bucket exceeds the CUDA grid limit");
  }
  TORCH_CHECK(descriptor_count == descriptors.size(0),
              "bucket counts do not match descriptors");

  TORCH_CHECK(total_small >= 0 && total_large >= 0,
              "workspace sizes must be non-negative");
  const int64_t matrix_count = cost.size(1);
  constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
  TORCH_CHECK(
      matrix_count == 0 || total_small <= kInt64Max / matrix_count,
      "small workspace size overflow");
  TORCH_CHECK(
      matrix_count == 0 || total_large <= kInt64Max / matrix_count,
      "large workspace size overflow");
  const int64_t small_elements = total_small * matrix_count;
  const int64_t large_elements = total_large * matrix_count;
  TORCH_CHECK(
      large_elements <= (kInt64Max - small_elements) / 3,
      "combined workspace size overflow");
}

}  // namespace

torch::Tensor solve(
    const torch::Tensor& cost,
    const torch::Tensor& descriptors,
    const std::vector<int64_t>& bucket_counts,
    int64_t total_small,
    int64_t total_large) {
  check_cuda_input(
      cost, descriptors, bucket_counts, total_small, total_large);
  return grouped_lap_cuda(
      cost, descriptors, bucket_counts, total_small, total_large);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "solve", &solve, "Internal segmented LAP (CUDA)",
      pybind11::arg("cost"), pybind11::arg("descriptors"),
      pybind11::arg("bucket_counts"), pybind11::arg("total_small"),
      pybind11::arg("total_large"));
}
