# Benchmark report

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti, 68 SMs, 11 GiB |
| Driver | 535.230.02 |
| PyTorch / CUDA | 2.0.1 / 11.8 |
| Reference | torch-linear-assignment v0.0.6 (`9c842e34`) |
| Timing | 10 warmups, 21 CUDA Event samples, median |
| Input | deterministic `torch.rand`, float32 and float64 |

The input to the segmented solver is `[sum(N_i), M, Y]`. The reference input
is produced from the same tensor by global padding. Metadata for both paths is
built before timing, and objectives are checked before each case.

The reported stages are:

- **segmented**: segmented solve only;
- **padding**: materialize `[K*M, max(N_i), Y]` with one indexed CUDA copy;
- **transpose**: required contiguous transpose when `max(N_i) > Y`;
- **upstream**: solve-only on an already padded and contiguous tensor;
- **padded E2E**: padding, transpose, upstream solve, dtype conversion and
  restoration to the packed output layout.

For the distribution sweep:

```text
f(n,Y) = min(n,Y)^2 * max(n,Y)
padding_ratio = K * max(N_i) / sum(N_i)
work_inflation = K * f(max(N_i),Y) / sum(f(N_i,Y))
```

## LAP count

Uniform `N_i=64`, `Y=64`, `M=2`. Padding ratio and work inflation are both 1.

| LAPs | f32 segmented ms | f32 upstream ms | f32 solve / E2E speedup | f64 segmented ms | f64 upstream ms | f64 solve / E2E speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.500 | 11.272 | 7.51x / 7.51x | 1.203 | 18.524 | 15.40x / 15.42x |
| 256 | 1.057 | 11.927 | 11.29x / 11.35x | 1.368 | 20.708 | 15.14x / 15.16x |
| 1,024 | 1.507 | 13.361 | 8.87x / 8.90x | 2.461 | 22.811 | 9.27x / 9.35x |
| 4,096 | 4.772 | 16.083 | 3.37x / 3.46x | 8.341 | 27.134 | 3.25x / 3.34x |
| 16,384 | 17.925 | 87.372 | 4.87x / 4.97x | 31.621 | 133.006 | 4.21x / 4.31x |

The gap narrows once the reference kernel has enough LAPs to expose its batch
parallelism, but remains measurable at 16,384 LAPs on this GPU.

## Problem size

Uniform 1,024 LAPs, `Y=64`, `M=2`. Padding ratio and work inflation remain 1.

| N | f32 solve / E2E speedup | f64 solve / E2E speedup |
|---:|---:|---:|
| 8 | 8.57x / 9.11x | 8.52x / 9.09x |
| 16 | 10.67x / 10.83x | 9.41x / 9.89x |
| 32 | 10.32x / 10.60x | 11.65x / 11.98x |
| 64 | 9.60x / 9.68x | 9.55x / 9.64x |
| 128 | 4.68x / 4.86x | 4.37x / 4.53x |
| 256 | 4.72x / 4.96x | 3.81x / 4.03x |
| 512 | 4.27x / 4.49x | 2.75x / 2.94x |

For `N>Y`, the reference solve-only path receives a physically transposed
contiguous tensor, while the segmented solver keeps the packed input and reads
it through a logical transpose. The transpose cost is included only in E2E.

## Segment-size distribution

1,024 LAPs, `M=2`, `Y=64`, `max(N_i)=512`. All profiles have the same padded
shape.

| Profile | N min/mean/max | Padding ratio | Work inflation | f32 solve / E2E speedup | f64 solve / E2E speedup |
|---|---:|---:|---:|---:|---:|
| uniform | 512/512.0/512 | 1.00x | 1.00x | 4.26x / 4.47x | 2.79x / 2.98x |
| mild | 108/142.8/512 | 3.58x | 3.58x | 23.69x / 24.13x | 20.15x / 20.62x |
| moderate | 5/15.6/512 | 32.85x | 65.22x | 90.52x / 91.57x | 118.89x / 119.73x |
| heavy-tailed | 1/3.2/512 | 158.20x | 272.12x | 76.25x / 76.92x | 90.91x / 91.85x |

Representative stage medians:

| dtype/profile | segmented | padding | transpose | upstream | padded E2E |
|---|---:|---:|---:|---:|---:|
| f32 uniform | 6.666 | 0.779 | 0.649 | 28.395 | 29.787 |
| f32 heavy-tailed | 1.196 | 0.240 | 0.639 | 91.197 | 92.005 |
| f64 uniform | 12.921 | 1.474 | 1.003 | 36.045 | 38.527 |
| f64 heavy-tailed | 1.598 | 0.469 | 1.006 | 145.269 | 146.774 |

`work_inflation` is a complexity proxy, not a speedup prediction. Crouse LAP
runtime depends on the augmenting paths, and dummy rows change those paths.
This is why the moderate and heavy-tailed results are not monotonic.

## Attribution

```text
uniform speedup
-> primarily cooperative intra-LAP GPU parallelism

imbalanced extra speedup
-> primarily avoiding global padding and redundant LAP work
```

Uniform cases have no padding or work inflation, so their speedup cannot come
from the segmented layout. In imbalanced cases, padding materialization is much
smaller than upstream solve time; most of the additional gain comes from not
solving dummy rows.

These results apply to the tested GPU and synthetic inputs. CUDA architecture,
contention, cost values and workload shape can change the result.

## Reproduce

```bash
python -m pip install --no-build-isolation torch-linear-assignment==0.0.6
python benchmarks/benchmark.py
```
