from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Dict, Optional, Sequence, Tuple, Union

import torch


IntegerVector = Union[torch.Tensor, Sequence[int]]

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_COST_DTYPES = {torch.float32, torch.float64}
_BACKEND = None


@dataclass(frozen=True)
class _CudaMetadata:
    descriptors: torch.Tensor
    bucket_counts: Tuple[int, int, int, int]
    total_small: int
    total_large: int


@dataclass(frozen=True)
class GroupPlan:
    """Reusable metadata for contiguous variable-sized groups."""

    offsets: Tuple[int, ...]
    sizes: Tuple[int, ...]
    _cuda_cache: Dict[Tuple[str, Optional[int], int], _CudaMetadata] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        offsets = _checked_python_integers(self.offsets, "offsets")
        sizes = _checked_python_integers(self.sizes, "sizes")
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "sizes", tuple(sizes))

        if len(offsets) != len(sizes) + 1:
            raise ValueError("offsets must have exactly num_groups + 1 entries")
        if not offsets or offsets[0] != 0:
            raise ValueError("offsets must start at zero")
        if any(size < 0 for size in sizes):
            raise ValueError("group sizes must be non-negative")

        expected = [0]
        for size in sizes:
            expected.append(expected[-1] + size)
        if offsets != expected:
            raise ValueError("offsets and sizes are inconsistent")

    @property
    def num_rows(self) -> int:
        return self.offsets[-1]

    @property
    def num_groups(self) -> int:
        return len(self.sizes)

    def _cuda_metadata(self, target_count: int, device: torch.device) -> _CudaMetadata:
        device = torch.device(device)
        key = (device.type, device.index, int(target_count))
        cached = self._cuda_cache.get(key)
        if cached is not None:
            return cached

        buckets = [[], [], [], []]
        total_small = 0
        total_large = 0
        for group_index, size in enumerate(self.sizes):
            if size and target_count:
                nr = min(size, target_count)
                nc = max(size, target_count)
                if nc <= 32:
                    bucket = 0
                elif nc <= 64:
                    bucket = 1
                elif nc <= 128:
                    bucket = 2
                else:
                    bucket = 3
                buckets[bucket].append(
                    (
                        self.offsets[group_index],
                        size,
                        total_small,
                        total_large,
                    )
                )
                total_small += nr
                total_large += nc

        bucket_counts = tuple(len(bucket) for bucket in buckets)
        descriptor_rows = [row for bucket in buckets for row in bucket]
        descriptors = torch.tensor(
            descriptor_rows, dtype=torch.int64, device=device
        ).reshape(-1, 4)

        metadata = _CudaMetadata(
            descriptors,
            bucket_counts,
            total_small,
            total_large,
        )
        self._cuda_cache[key] = metadata
        return metadata


def _as_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _checked_python_integers(values: Sequence[int], name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an integer vector")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name}[{index}] must be an integer")
        result.append(int(value))
    return result


def _check_integer_tensor(values: torch.Tensor, name: str) -> None:
    if values.layout != torch.strided:
        raise TypeError(f"{name} must use the strided tensor layout")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must have an integer dtype")


def _integer_values(values: IntegerVector, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        _check_integer_tensor(values, name)
        return values.detach().cpu().tolist()
    return _checked_python_integers(values, name)


def _apply_num_groups(
    sizes: list[int], num_groups: Optional[int], inferred_name: str
) -> list[int]:
    if num_groups is None:
        return sizes
    group_count = _as_nonnegative_int(num_groups, "num_groups")
    if group_count < len(sizes):
        raise ValueError(
            f"num_groups is smaller than the number inferred from {inferred_name}"
        )
    return sizes + [0] * (group_count - len(sizes))


def _plan_from_sizes(
    sizes: IntegerVector, num_groups: Optional[int]
) -> GroupPlan:
    checked = _integer_values(sizes, "sizes")
    if any(size < 0 for size in checked):
        raise ValueError("sizes must be non-negative")
    checked = _apply_num_groups(checked, num_groups, "sizes")
    offsets = [0]
    for size in checked:
        offsets.append(offsets[-1] + size)
    return GroupPlan(tuple(offsets), tuple(checked))


def _plan_from_offsets(
    offsets: IntegerVector, num_groups: Optional[int]
) -> GroupPlan:
    checked = _integer_values(offsets, "offsets")
    if not checked or checked[0] != 0:
        raise ValueError("offsets must contain at least [0] and start at zero")
    if any(right < left for left, right in zip(checked, checked[1:])):
        raise ValueError("offsets must be monotonically nondecreasing")
    sizes = [right - left for left, right in zip(checked, checked[1:])]
    sizes = _apply_num_groups(sizes, num_groups, "offsets")
    return _plan_from_sizes(sizes, num_groups=None)


def _compressed_group_runs(groups: IntegerVector) -> Tuple[list[int], list[int]]:
    if isinstance(groups, torch.Tensor):
        _check_integer_tensor(groups, "groups")
        if groups.numel() == 0:
            return [], []
        labels, counts = torch.unique_consecutive(
            groups.detach(), return_counts=True
        )
        compact = torch.stack((labels.to(torch.int64), counts))
        label_values, count_values = compact.cpu().tolist()
        return label_values, count_values

    values = _checked_python_integers(groups, "groups")
    if not values:
        return [], []
    labels = [values[0]]
    counts = [1]
    for value in values[1:]:
        if value == labels[-1]:
            counts[-1] += 1
        else:
            labels.append(value)
            counts.append(1)
    return labels, counts


def _plan_from_groups(
    groups: IntegerVector, num_groups: Optional[int]
) -> GroupPlan:
    labels, counts = _compressed_group_runs(groups)
    if any(label < 0 for label in labels):
        raise ValueError("groups must contain only non-negative labels")
    if any(right <= left for left, right in zip(labels, labels[1:])):
        raise ValueError(
            "groups must be monotonically nondecreasing so each group is contiguous"
        )

    inferred = labels[-1] + 1 if labels else 0
    group_count = (
        inferred
        if num_groups is None
        else _as_nonnegative_int(num_groups, "num_groups")
    )
    if group_count < inferred:
        raise ValueError("num_groups is smaller than the largest group label plus one")

    sizes = [0] * group_count
    for label, count in zip(labels, counts):
        sizes[label] = count
    return _plan_from_sizes(sizes, num_groups=None)


def _resolve_plan(
    *,
    group_plan: Optional[GroupPlan],
    offsets: Optional[IntegerVector],
    groups: Optional[IntegerVector],
    sizes: Optional[IntegerVector],
    num_groups: Optional[int],
) -> GroupPlan:
    if group_plan is not None:
        if not isinstance(group_plan, GroupPlan):
            raise TypeError("group_plan must be a GroupPlan or None")
        return group_plan

    sources = [offsets is not None, groups is not None, sizes is not None]
    if sum(sources) != 1:
        raise ValueError(
            "provide exactly one of offsets, groups, or sizes when group_plan is None"
        )
    if offsets is not None:
        return _plan_from_offsets(offsets, num_groups)
    if groups is not None:
        return _plan_from_groups(groups, num_groups)
    assert sizes is not None
    return _plan_from_sizes(sizes, num_groups)


def _check_cost(cost: torch.Tensor) -> None:
    if not isinstance(cost, torch.Tensor):
        raise TypeError("cost must be a torch.Tensor")
    if cost.device.type != "cuda":
        raise ValueError("cost must be a CUDA tensor")
    if cost.ndim != 3:
        raise ValueError("cost must have shape [N, M, Y]")
    if cost.layout != torch.strided:
        raise TypeError("cost must use the strided tensor layout")
    if cost.dtype not in _COST_DTYPES:
        raise TypeError("cost must have dtype torch.float32 or torch.float64")


def _load_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from . import _C
    except ImportError as error:
        raise RuntimeError(
            "segmented_linear_assignment extension is not built. Run "
            "`python -m pip install --no-build-isolation .` first."
        ) from error
    _BACKEND = _C
    return _BACKEND


def linear_assignment(
    cost: torch.Tensor,
    *,
    group_plan: Optional[GroupPlan] = None,
    offsets: Optional[IntegerVector] = None,
    groups: Optional[IntegerVector] = None,
    sizes: Optional[IntegerVector] = None,
    num_groups: Optional[int] = None,
) -> Tuple[torch.Tensor, GroupPlan]:
    """Solve one LAP per group and M slice, returning result and plan.

    ``cost`` must be a CUDA tensor with shape ``[N, M, Y]``. Supply a cached
    ``group_plan`` when available; otherwise supply exactly one of ``offsets``,
    ``groups`` (one contiguous label per row), or ``sizes``. ``num_groups`` is
    inferred when omitted and can add trailing empty groups when provided.
    """

    _check_cost(cost)
    plan = _resolve_plan(
        group_plan=group_plan,
        offsets=offsets,
        groups=groups,
        sizes=sizes,
        num_groups=num_groups,
    )
    if plan.num_rows != cost.size(0):
        raise ValueError(
            f"group metadata has {plan.num_rows} rows, but cost has {cost.size(0)}"
        )

    if cost.size(0) == 0 or cost.size(1) == 0 or cost.size(2) == 0:
        return torch.full(
            (cost.size(0), cost.size(1)),
            -1,
            dtype=torch.int64,
            device=cost.device,
        ), plan

    backend = _load_backend()
    metadata = plan._cuda_metadata(cost.size(2), cost.device)
    cost = cost.resolve_neg()
    return backend.solve(
        cost,
        metadata.descriptors,
        metadata.bucket_counts,
        metadata.total_small,
        metadata.total_large,
    ), plan
