/* Crouse LAP adapted from torch-linear-assignment for packed segments. */

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t read_cost(
    const scalar_t* cost,
    int64_t group_start,
    int solver_row,
    int solver_column,
    int matrix_index,
    int64_t stride_row,
    int64_t stride_matrix,
    int64_t stride_column,
    bool transposed) {
  const int64_t input_row =
      group_start + (transposed ? solver_column : solver_row);
  const int64_t input_column = transposed ? solver_row : solver_column;
  return cost[input_row * stride_row +
              static_cast<int64_t>(matrix_index) * stride_matrix +
              input_column * stride_column];
}

// Match upstream ties: prefer free columns; free/free takes the later
// remaining position, matched/matched the earlier one.
template <typename scalar_t>
__device__ __forceinline__ bool better_candidate(
    int candidate_position,
    int best_position,
    const int* remaining,
    const scalar_t* shortest_path_costs,
    const int* row4col,
    scalar_t infinity) {
  if (candidate_position < 0) {
    return false;
  }

  const int candidate_column = remaining[candidate_position];
  const scalar_t candidate_value = shortest_path_costs[candidate_column];
  const bool candidate_is_free = row4col[candidate_column] == -1;

  if (best_position < 0) {
    return candidate_value < infinity ||
           (candidate_value == infinity && candidate_is_free);
  }

  const int best_column = remaining[best_position];
  const scalar_t best_value = shortest_path_costs[best_column];
  if (candidate_value < best_value) {
    return true;
  }
  if (candidate_value == best_value) {
    const bool best_is_free = row4col[best_column] == -1;
    if (candidate_is_free != best_is_free) {
      return candidate_is_free;
    }
    return candidate_is_free ? candidate_position > best_position
                             : candidate_position < best_position;
  }
  return false;
}

template <typename scalar_t, int kBlockSize>
__global__ void lap_kernel(
    const scalar_t* __restrict__ cost,
    const int64_t* __restrict__ descriptors,
    int matrix_count,
    int target_count,
    int64_t cost_stride_row,
    int64_t cost_stride_matrix,
    int64_t cost_stride_column,
    scalar_t* __restrict__ u_all,
    scalar_t* __restrict__ v_all,
    scalar_t* __restrict__ shortest_all,
    int* __restrict__ path_all,
    int* __restrict__ col4row_all,
    int* __restrict__ row4col_all,
    int* __restrict__ remaining_all,
    uint8_t* __restrict__ visited_rows_all,
    uint8_t* __restrict__ visited_columns_all,
    int64_t* __restrict__ assignment) {
  static_assert(kBlockSize >= 32 && kBlockSize <= 256,
                "unsupported CUDA block size");
  static_assert((kBlockSize & (kBlockSize - 1)) == 0,
                "CUDA block size must be a power of two");

  constexpr int kWarpSize = 32;
  constexpr int kWarpCount = kBlockSize / kWarpSize;
  __shared__ int reduction_positions[kWarpCount];
  __shared__ scalar_t min_val;
  __shared__ int matrix_index;
  __shared__ int nr;
  __shared__ int nc;
  __shared__ int current_row;
  __shared__ int sink;
  __shared__ int num_remaining;
  __shared__ int failed;
  __shared__ int transposed;
  __shared__ int64_t group_start;
  __shared__ int64_t small_base;
  __shared__ int64_t large_base;

  const int tid = threadIdx.x;
  const int task_index = static_cast<int>(blockIdx.x);
  const int descriptor_index = task_index / matrix_count;

  if (tid == 0) {
    matrix_index = task_index - descriptor_index * matrix_count;
    const int64_t* descriptor =
        descriptors + static_cast<int64_t>(descriptor_index) * 4;
    group_start = descriptor[0];
    const int group_size = static_cast<int>(descriptor[1]);
    nr = min(group_size, target_count);
    nc = max(group_size, target_count);
    transposed = group_size > target_count;
    small_base = descriptor[2] * matrix_count +
                 static_cast<int64_t>(matrix_index) * nr;
    large_base = descriptor[3] * matrix_count +
                 static_cast<int64_t>(matrix_index) * nc;
    failed = 0;
  }
  __syncthreads();

  scalar_t* const u = u_all + small_base;
  scalar_t* const v = v_all + large_base;
  scalar_t* const shortest_path_costs = shortest_all + large_base;
  int* const path = path_all + large_base;
  int* const col4row = col4row_all + small_base;
  int* const row4col = row4col_all + large_base;
  int* const remaining = remaining_all + large_base;
  uint8_t* const visited_rows = visited_rows_all + small_base;
  uint8_t* const visited_columns = visited_columns_all + large_base;
  const scalar_t infinity = std::numeric_limits<scalar_t>::infinity();

  for (int i = tid; i < nr; i += kBlockSize) {
    u[i] = static_cast<scalar_t>(0);
    col4row[i] = -1;
  }
  for (int j = tid; j < nc; j += kBlockSize) {
    v[j] = static_cast<scalar_t>(0);
    path[j] = -1;
    row4col[j] = -1;
  }
  __syncthreads();

  for (int cur_row = 0; cur_row < nr; ++cur_row) {
    for (int i = tid; i < nr; i += kBlockSize) {
      visited_rows[i] = 0;
    }
    for (int it = tid; it < nc; it += kBlockSize) {
      visited_columns[it] = 0;
      remaining[it] = nc - it - 1;
      shortest_path_costs[it] = infinity;
    }
    if (tid == 0) {
      min_val = static_cast<scalar_t>(0);
      num_remaining = nc;
      current_row = cur_row;
      sink = -1;
    }
    __syncthreads();

    while (sink == -1 && failed == 0) {
      if (tid == 0) {
        visited_rows[current_row] = 1;
      }
      __syncthreads();

      // Keep the upstream reduced-cost evaluation order.
      const scalar_t base_r = min_val - u[current_row];
      int local_position = -1;
      for (int it = tid; it < num_remaining; it += kBlockSize) {
        const int j = remaining[it];
        const scalar_t r =
            base_r +
                read_cost(
                    cost, group_start, current_row, j, matrix_index,
                    cost_stride_row, cost_stride_matrix, cost_stride_column,
                    transposed != 0) -
            v[j];
        if (r < shortest_path_costs[j]) {
          path[j] = current_row;
          shortest_path_costs[j] = r;
        }
        if (better_candidate(
                it, local_position, remaining, shortest_path_costs, row4col,
                infinity)) {
          local_position = it;
        }
      }

      // Reduce positions only; floating-point values are never combined.
      __syncwarp();
      constexpr unsigned int kFullWarpMask = 0xffffffffU;
      const int lane = tid & (kWarpSize - 1);
      const int warp = tid / kWarpSize;
      int candidate = local_position;
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const int other =
            __shfl_down_sync(kFullWarpMask, candidate, offset);
        if (lane < offset &&
            better_candidate(
                other, candidate, remaining, shortest_path_costs, row4col,
                infinity)) {
          candidate = other;
        }
      }
      if (lane == 0) {
        reduction_positions[warp] = candidate;
      }
      __syncthreads();

      if (warp == 0) {
        candidate = lane < kWarpCount ? reduction_positions[lane] : -1;
        for (int offset = kWarpCount / 2; offset > 0; offset >>= 1) {
          const int other =
              __shfl_down_sync(kFullWarpMask, candidate, offset);
          if (lane < offset &&
              better_candidate(
                  other, candidate, remaining, shortest_path_costs, row4col,
                  infinity)) {
            candidate = other;
          }
        }
        if (lane == 0) {
          reduction_positions[0] = candidate;
        }
      }
      __syncthreads();

      if (tid == 0) {
        const int selected_position = reduction_positions[0];
        if (selected_position < 0) {
          failed = 1;
        } else {
          const int j = remaining[selected_position];
          min_val = shortest_path_costs[j];
          if (min_val == infinity) {
            failed = 1;
          } else {
            if (row4col[j] == -1) {
              sink = j;
            } else {
              current_row = row4col[j];
            }
            visited_columns[j] = 1;
            remaining[selected_position] = remaining[--num_remaining];
          }
        }
      }
      __syncthreads();
    }

    if (failed) {
      if (tid == 0) {
        CUDA_KERNEL_ASSERT(false && "Infeasible matrix");
      }
      return;
    }

    if (tid == 0) {
      u[cur_row] += min_val;
    }
    for (int i = tid; i < nr; i += kBlockSize) {
      if (visited_rows[i] && i != cur_row) {
        u[i] += min_val - shortest_path_costs[col4row[i]];
      }
    }
    for (int j = tid; j < nc; j += kBlockSize) {
      if (visited_columns[j]) {
        v[j] -= min_val - shortest_path_costs[j];
      }
    }
    __syncthreads();

    if (tid == 0) {
      int i = -1;
      int j = sink;
      while (i != cur_row) {
        i = path[j];
        row4col[j] = i;
        const int previous_column = j;
        j = col4row[i];
        col4row[i] = previous_column;
      }
    }
    __syncthreads();
  }

  if (!transposed) {
    for (int i = tid; i < nr; i += kBlockSize) {
      assignment[(group_start + i) * matrix_count + matrix_index] =
          static_cast<int64_t>(col4row[i]);
    }
  } else {
    for (int input_row = tid; input_row < nc; input_row += kBlockSize) {
      assignment[(group_start + input_row) * matrix_count + matrix_index] =
          static_cast<int64_t>(row4col[input_row]);
    }
  }
}

template <typename scalar_t, int kBlockSize>
void launch_bucket(
    const torch::Tensor& cost,
    const torch::Tensor& descriptors,
    int64_t descriptor_offset,
    int64_t descriptor_count,
    scalar_t* u,
    scalar_t* v,
    scalar_t* shortest_path_costs,
    int* path,
    int* col4row,
    int* row4col,
    int* remaining,
    uint8_t* visited_rows,
    uint8_t* visited_columns,
    torch::Tensor& assignment) {
  if (descriptor_count == 0) {
    return;
  }

  const int matrix_count = static_cast<int>(cost.size(1));
  const int64_t task_count = descriptor_count * matrix_count;
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(cost.get_device());
  lap_kernel<scalar_t, kBlockSize>
      <<<static_cast<unsigned int>(task_count), kBlockSize, 0,
         stream.stream()>>>(
          cost.data_ptr<scalar_t>(),
          descriptors.data_ptr<int64_t>() + descriptor_offset * 4,
          matrix_count, static_cast<int>(cost.size(2)), cost.stride(0),
          cost.stride(1), cost.stride(2), u, v,
          shortest_path_costs, path, col4row, row4col, remaining,
          visited_rows, visited_columns, assignment.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_buckets(
    const torch::Tensor& cost,
    const torch::Tensor& descriptors,
    const std::vector<int64_t>& bucket_counts,
    int64_t total_small,
    int64_t total_large,
    torch::Tensor& assignment) {
  const int64_t matrix_count = cost.size(1);
  const int64_t small_elements = total_small * matrix_count;
  const int64_t large_elements = total_large * matrix_count;

  torch::Tensor scalar_workspace =
      torch::empty({small_elements + 2 * large_elements}, cost.options());
  torch::Tensor integer_workspace = torch::empty(
      {small_elements + 3 * large_elements},
      cost.options().dtype(torch::kInt));
  torch::Tensor byte_workspace = torch::empty(
      {small_elements + large_elements},
      cost.options().dtype(torch::kUInt8));

  scalar_t* const scalar_base = scalar_workspace.data_ptr<scalar_t>();
  scalar_t* const u = scalar_base;
  scalar_t* const v = u + small_elements;
  scalar_t* const shortest_path_costs = v + large_elements;

  int* const integer_base = integer_workspace.data_ptr<int>();
  int* const path = integer_base;
  int* const col4row = path + large_elements;
  int* const row4col = col4row + small_elements;
  int* const remaining = row4col + large_elements;

  uint8_t* const byte_base = byte_workspace.data_ptr<uint8_t>();
  uint8_t* const visited_rows = byte_base;
  uint8_t* const visited_columns = visited_rows + small_elements;

  int64_t offset = 0;
  launch_bucket<scalar_t, 32>(
      cost, descriptors, offset, bucket_counts[0], u, v,
      shortest_path_costs, path, col4row, row4col, remaining, visited_rows,
      visited_columns, assignment);
  offset += bucket_counts[0];
  launch_bucket<scalar_t, 64>(
      cost, descriptors, offset, bucket_counts[1], u, v,
      shortest_path_costs, path, col4row, row4col, remaining, visited_rows,
      visited_columns, assignment);
  offset += bucket_counts[1];
  launch_bucket<scalar_t, 128>(
      cost, descriptors, offset, bucket_counts[2], u, v,
      shortest_path_costs, path, col4row, row4col, remaining, visited_rows,
      visited_columns, assignment);
  offset += bucket_counts[2];
  launch_bucket<scalar_t, 256>(
      cost, descriptors, offset, bucket_counts[3], u, v,
      shortest_path_costs, path, col4row, row4col, remaining, visited_rows,
      visited_columns, assignment);
}

}  // namespace

torch::Tensor grouped_lap_cuda(
    const torch::Tensor& cost,
    const torch::Tensor& descriptors,
    const std::vector<int64_t>& bucket_counts,
    int64_t total_small,
    int64_t total_large) {
  const c10::cuda::CUDAGuard device_guard(cost.device());
  if (descriptors.size(0) == 0 || cost.size(0) == 0 ||
      cost.size(1) == 0 || cost.size(2) == 0) {
    return torch::full(
        {cost.size(0), cost.size(1)}, -1,
        cost.options().dtype(torch::kLong));
  }

  torch::Tensor assignment = torch::empty(
      {cost.size(0), cost.size(1)}, cost.options().dtype(torch::kLong));
  if (cost.scalar_type() == torch::kFloat) {
    launch_buckets<float>(
        cost, descriptors, bucket_counts, total_small, total_large, assignment);
  } else {
    launch_buckets<double>(
        cost, descriptors, bucket_counts, total_small, total_large, assignment);
  }
  return assignment;
}
