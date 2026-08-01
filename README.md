# segmented-linear-assignment

CUDA linear assignment for variable-size segmented batches.

The input cost tensor has shape `[sum(N_i), M, Y]`. Each contiguous segment
`[N_i, Y]` and each `M` slice is solved as an independent LAP. Costs and
workspace stay packed; the operator does not build a
`[K, M, max(N_i), Y]` tensor.

The CUDA kernel follows the rectangular Crouse shortest augmenting path
algorithm used by
[torch-linear-assignment v0.0.6](https://github.com/ivan-chai/torch-linear-assignment/tree/9c842e34f29d55c80f4529bf62f520eed1048442).
Float32 and float64 inputs keep their dtype throughout the solver. When
`N_i > Y`, the problem is transposed internally and unmatched input rows are
returned as `-1`.

## Install

Requirements:

- Python 3.8+
- CUDA-enabled PyTorch 2.0+
- an NVIDIA GPU
- an NVCC toolkit matching `torch.version.cuda`

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
python -m pip install --no-build-isolation --no-deps .
```

Set `CUDAHOSTCXX` when NVCC needs an explicit host compiler. Headless builds
also need `TORCH_CUDA_ARCH_LIST` set to the target compute capabilities.

## Usage

```python
import torch
from segmented_linear_assignment import linear_assignment

sizes = [3, 5, 2]
cost = torch.rand(sum(sizes), 4, 7, device="cuda")

assignment, plan = linear_assignment(cost, sizes=sizes)
```

`assignment` is an int64 CUDA tensor with shape `[sum(N_i), M]`. Values are
local column indices in `[0, Y)`, or `-1` for unmatched rows.

Segment metadata can be supplied as `sizes`, `offsets`, or monotonically
increasing contiguous `groups`. Reuse the returned plan when the segmentation
does not change:

```python
next_assignment, _ = linear_assignment(next_cost, group_plan=plan)
```

## Benchmark

```bash
python benchmarks/benchmark.py
```

The benchmark compares the packed solver with
`torch-linear-assignment + global padding`. It measures segmented solve,
padding, required transpose, upstream solve-only, and padded end-to-end with
CUDA Events.

For uniform sizes, the main gain comes from cooperative intra-LAP GPU
parallelism. Imbalanced batches gain additionally by avoiding padded rows and
redundant LAP work. Experimental setup and measured values are in the
[benchmark report](benchmarks/BENCHMARK_REPORT.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
