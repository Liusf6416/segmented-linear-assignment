#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch

try:
    from segmented_linear_assignment import GroupPlan, linear_assignment
except ModuleNotFoundError as error:
    if error.name != "segmented_linear_assignment":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.assignment import GroupPlan, linear_assignment


LAP_COUNTS = (64, 256, 1024, 4096, 16384)
ROW_SIZES = (8, 16, 32, 64, 128, 256, 512)
PROFILES = (
    ("uniform", 0.0),
    ("mild", 0.25),
    ("moderate", 0.75),
    ("heavy-tailed", 1.5),
)
M = 2
Y = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("all", "lap-count", "size", "distribution"),
        default="all",
    )
    parser.add_argument(
        "--dtype", choices=("all", "float32", "float64"), default="all"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup and --repeats must be positive")
    return args


def make_group_plan(sizes: Sequence[int]) -> GroupPlan:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return GroupPlan(tuple(offsets), tuple(sizes))


@dataclass(frozen=True)
class PaddingPlan:
    sizes: Tuple[int, ...]
    group_index: torch.Tensor
    local_row: torch.Tensor

    @classmethod
    def build(cls, sizes: Sequence[int], device: torch.device) -> "PaddingPlan":
        sizes = tuple(sizes)
        sizes_cpu = torch.tensor(sizes, dtype=torch.int64)
        groups = torch.repeat_interleave(
            torch.arange(len(sizes), dtype=torch.int64), sizes_cpu
        )
        offsets = torch.zeros(len(sizes), dtype=torch.int64)
        if len(sizes) > 1:
            offsets[1:] = sizes_cpu.cumsum(0)[:-1]
        rows = torch.arange(sum(sizes), dtype=torch.int64) - offsets[groups]
        return cls(sizes, groups.to(device), rows.to(device))

    @property
    def maximum(self) -> int:
        return max(self.sizes)

    def pad(self, cost: torch.Tensor) -> torch.Tensor:
        padded = torch.full(
            (len(self.sizes), M, self.maximum, Y),
            float(Y + 1),
            dtype=cost.dtype,
            device=cost.device,
        )
        padded[self.group_index, :, self.local_row, :] = cost
        return padded.view(len(self.sizes) * M, self.maximum, Y)

    def unpack(self, assignment: torch.Tensor) -> torch.Tensor:
        assignment = assignment.view(len(self.sizes), M, self.maximum)
        return assignment[self.group_index, :, self.local_row].contiguous()


class Upstream:
    def __init__(self) -> None:
        try:
            from torch_linear_assignment import _backend
        except ImportError as error:
            raise SystemExit("torch-linear-assignment is required") from error
        if not _backend.has_cuda():
            raise SystemExit("torch-linear-assignment has no CUDA backend")
        self.backend = _backend

    @staticmethod
    def prepare(cost: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if cost.size(1) > cost.size(2):
            return cost.transpose(1, 2).contiguous(), True
        return cost, False

    def solve(self, cost: torch.Tensor, transposed: bool) -> torch.Tensor:
        if not cost.is_contiguous():
            raise ValueError("upstream input must be contiguous")
        col4row, row4col = self.backend.batch_linear_assignment(cost)
        return row4col if transposed else col4row


def cuda_median(
    operation: Callable[[], object], warmup: int, repeats: int
) -> float:
    result: object = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()

    events = []
    for _ in range(repeats):
        result = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    del result
    return float(
        statistics.median(start.elapsed_time(end) for start, end in events)
    )


def make_cost(
    sizes: Sequence[int], dtype: torch.dtype, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.rand(
        (sum(sizes), M, Y),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )


def objectives(
    cost: torch.Tensor, assignment: torch.Tensor, plan: GroupPlan
) -> torch.Tensor:
    columns = assignment.clamp_min(0)
    selected = cost.gather(2, columns.unsqueeze(-1)).squeeze(-1)
    selected = selected.masked_fill(assignment < 0, 0).double()
    prefix = torch.cat(
        (
            torch.zeros((1, M), dtype=torch.float64, device=cost.device),
            selected.cumsum(0),
        )
    )
    offsets = torch.tensor(plan.offsets, dtype=torch.int64, device=cost.device)
    return prefix[offsets[1:]] - prefix[offsets[:-1]]


def validate(
    cost: torch.Tensor,
    plan: GroupPlan,
    segmented: torch.Tensor,
    padded: torch.Tensor,
) -> float:
    left = objectives(cost, segmented, plan)
    right = objectives(cost, padded, plan)
    error = float((left - right).abs().max().item())
    tolerance = 2e-5 if cost.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
    return error


def lap_work(n: int) -> int:
    return min(n, Y) ** 2 * max(n, Y)


def power_law_sizes(
    groups: int, maximum: int, exponent: float, seed: int
) -> Tuple[int, ...]:
    if exponent == 0:
        return (maximum,) * groups
    sizes = [
        max(1, round(maximum / math.pow(rank, exponent)))
        for rank in range(1, groups + 1)
    ]
    random.Random(seed).shuffle(sizes)
    return tuple(sizes)


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    sizes: Tuple[int, ...]

    @property
    def laps(self) -> int:
        return len(self.sizes) * M


def make_cases(suite: str, seed: int) -> List[Case]:
    cases = []
    if suite in ("all", "lap-count"):
        cases.extend(
            Case("lap-count", str(laps), (64,) * (laps // M))
            for laps in LAP_COUNTS
        )
    if suite in ("all", "size"):
        cases.extend(
            Case("size", str(n), (n,) * (1024 // M)) for n in ROW_SIZES
        )
    if suite in ("all", "distribution"):
        groups = 1024 // M
        cases.extend(
            Case(
                "distribution",
                name,
                power_law_sizes(groups, 512, exponent, seed + index),
            )
            for index, (name, exponent) in enumerate(PROFILES)
        )
    return cases


def benchmark_case(
    case: Case,
    dtype_name: str,
    seed: int,
    warmup: int,
    repeats: int,
    upstream: Upstream,
) -> Dict[str, object]:
    dtype = getattr(torch, dtype_name)
    cost = make_cost(case.sizes, dtype, seed)
    group_plan = make_group_plan(case.sizes)
    padding_plan = PaddingPlan.build(case.sizes, cost.device)

    segmented_ready, _ = linear_assignment(cost, group_plan=group_plan)
    padded_ready = padding_plan.pad(cost)
    prepared_ready, transposed = upstream.prepare(padded_ready)
    upstream_ready = upstream.solve(prepared_ready, transposed)
    restored_ready = padding_plan.unpack(upstream_ready.long())
    objective_error = validate(
        cost, group_plan, segmented_ready, restored_ready
    )
    torch.cuda.synchronize()

    def segmented_solve() -> torch.Tensor:
        return linear_assignment(cost, group_plan=group_plan)[0]

    def padding() -> torch.Tensor:
        return padding_plan.pad(cost)

    def transpose_contiguous() -> torch.Tensor:
        return upstream.prepare(padded_ready)[0]

    def solve_only() -> torch.Tensor:
        return upstream.solve(prepared_ready, transposed)

    def end_to_end() -> torch.Tensor:
        padded = padding_plan.pad(cost)
        prepared, is_transposed = upstream.prepare(padded)
        assignment = upstream.solve(prepared, is_transposed)
        return padding_plan.unpack(assignment.long())

    segmented_ms = cuda_median(segmented_solve, warmup, repeats)
    padding_ms = cuda_median(padding, warmup, repeats)
    transpose_ms = (
        cuda_median(transpose_contiguous, warmup, repeats) if transposed else 0.0
    )
    upstream_ms = cuda_median(solve_only, warmup, repeats)
    end_to_end_ms = cuda_median(end_to_end, warmup, repeats)

    groups = len(case.sizes)
    maximum = max(case.sizes)
    actual_work = sum(lap_work(n) for n in case.sizes)
    padded_work = groups * lap_work(maximum)
    return {
        "suite": case.suite,
        "case": case.name,
        "dtype": dtype_name,
        "laps": case.laps,
        "min_n": min(case.sizes),
        "mean_n": sum(case.sizes) / groups,
        "max_n": maximum,
        "padding_ratio": groups * maximum / sum(case.sizes),
        "work_inflation": padded_work / actual_work,
        "objective_error": objective_error,
        "segmented_ms": segmented_ms,
        "padding_ms": padding_ms,
        "transpose_ms": transpose_ms,
        "upstream_ms": upstream_ms,
        "end_to_end_ms": end_to_end_ms,
        "solve_speedup": upstream_ms / segmented_ms,
        "end_to_end_speedup": end_to_end_ms / segmented_ms,
    }


def print_results(results: Sequence[Dict[str, object]]) -> None:
    print(
        "suite         case          dtype     LAPs  N[min/mean/max] "
        " padR  workR  seg(ms)  pad(ms)  xpose(ms)  upstream(ms) "
        " e2e(ms)  solveX  e2eX"
    )
    for row in results:
        n_range = (
            f"{row['min_n']}/{float(row['mean_n']):.1f}/{row['max_n']}"
        )
        print(
            f"{row['suite']:<13} {row['case']:<13} {row['dtype']:<8} "
            f"{row['laps']:>6} {n_range:>17} "
            f"{row['padding_ratio']:>5.2f} {row['work_inflation']:>6.2f} "
            f"{row['segmented_ms']:>8.3f} {row['padding_ms']:>8.3f} "
            f"{row['transpose_ms']:>10.3f} {row['upstream_ms']:>13.3f} "
            f"{row['end_to_end_ms']:>8.3f} {row['solve_speedup']:>6.2f}x "
            f"{row['end_to_end_speedup']:>5.2f}x"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtypes = ("float32", "float64") if args.dtype == "all" else (args.dtype,)
    cases = make_cases(args.suite, args.seed)
    upstream = Upstream()
    results = []
    for dtype_index, dtype in enumerate(dtypes):
        for case_index, case in enumerate(cases):
            print(f"running {dtype} {case.suite} {case.name}", flush=True)
            results.append(
                benchmark_case(
                    case,
                    dtype,
                    args.seed + dtype_index * 100000 + case_index,
                    args.warmup,
                    args.repeats,
                    upstream,
                )
            )

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        f"device={device.name} SMs={device.multi_processor_count} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"warmup={args.warmup} repeats={args.repeats} statistic=median"
    )
    print_results(results)

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
