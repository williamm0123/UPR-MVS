from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer.attention_backend import attention_forward


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class BenchmarkResult:
    mode: str
    backend: str
    metric_name: str
    elapsed_ms: float
    batches_per_s: float
    samples_per_s: float
    max_mem_mb: float
    speedup_vs_eager: float
    mem_delta_mb_vs_eager: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager vs SDPA attention backends.")
    parser.add_argument("--device", type=str, default="cuda", help="Benchmark device. Use cuda on A100.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=sorted(DTYPE_MAP), help="Tensor dtype.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--query-len", type=int, default=1024)
    parser.add_argument("--key-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save benchmark results as JSON.")
    parser.add_argument("--output-csv", type=str, default="", help="Optional path to save benchmark results as CSV.")
    return parser.parse_args()


def build_qkv(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    q = torch.randn(
        args.batch_size,
        args.num_heads,
        args.query_len,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        args.batch_size,
        args.num_heads,
        args.key_len,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        args.batch_size,
        args.num_heads,
        args.key_len,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return q, k, v


def clone_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_clone = q.detach().clone().requires_grad_(requires_grad)
    k_clone = k.detach().clone().requires_grad_(requires_grad)
    v_clone = v.detach().clone().requires_grad_(requires_grad)
    return q_clone, k_clone, v_clone


def run_attention_step(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    training: bool,
    dropout_p: float,
) -> None:
    if training:
        out = attention_forward(q, k, v, backend=backend, training=True, dropout_p=dropout_p)
        loss = out.float().square().mean()
        loss.backward()
        q.grad = None
        k.grad = None
        v.grad = None
        return

    with torch.inference_mode():
        attention_forward(q, k, v, backend=backend, training=False, dropout_p=dropout_p)


def benchmark_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    training: bool,
    dropout_p: float,
    warmup_iters: int,
    iters: int,
) -> tuple[float, float]:
    q_run, k_run, v_run = clone_qkv(q, k, v, requires_grad=training)
    for _ in range(warmup_iters):
        run_attention_step(backend, q_run, k_run, v_run, training=training, dropout_p=dropout_p)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(q.device)
    start = time.perf_counter()
    for _ in range(iters):
        run_attention_step(backend, q_run, k_run, v_run, training=training, dropout_p=dropout_p)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
    max_mem_mb = torch.cuda.max_memory_allocated(q.device) / (1024.0 * 1024.0)
    del q_run, k_run, v_run
    return elapsed_ms, max_mem_mb


def _validate_pair(eager_out: torch.Tensor, sdpa_out: torch.Tensor, *, mode_name: str) -> None:
    if eager_out.shape != sdpa_out.shape:
        raise RuntimeError(f"{mode_name} output shape mismatch: eager={tuple(eager_out.shape)} vs sdpa={tuple(sdpa_out.shape)}")
    if not torch.isfinite(eager_out).all():
        raise RuntimeError(f"Eager attention produced NaN/Inf outputs in {mode_name}.")
    if not torch.isfinite(sdpa_out).all():
        raise RuntimeError(f"SDPA attention produced NaN/Inf outputs in {mode_name}.")


def compare_deterministic_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    training: bool,
    dropout_p: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    eager_out = attention_forward(q, k, v, backend="eager", training=training, dropout_p=dropout_p)
    torch.manual_seed(seed)
    sdpa_out = attention_forward(q, k, v, backend="sdpa", training=training, dropout_p=dropout_p)
    _validate_pair(eager_out, sdpa_out, mode_name="deterministic_compare")
    return (eager_out - sdpa_out).abs().max().item()


def validate_configured_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    training: bool,
    dropout_p: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    eager_out = attention_forward(q, k, v, backend="eager", training=training, dropout_p=dropout_p)
    torch.manual_seed(seed)
    sdpa_out = attention_forward(q, k, v, backend="sdpa", training=training, dropout_p=dropout_p)
    _validate_pair(eager_out, sdpa_out, mode_name="configured_run")


def build_result_rows(
    *,
    mode_name: str,
    batch_size: int,
    backend_timings: dict[str, tuple[float, float]],
) -> list[BenchmarkResult]:
    eager_elapsed_ms, eager_mem_mb = backend_timings["eager"]
    metric_name = "attn_step_ms" if mode_name == "train" else "attn_forward_ms"
    rows: list[BenchmarkResult] = []
    for backend in ("eager", "sdpa", "auto"):
        elapsed_ms, max_mem_mb = backend_timings[backend]
        batches_per_s = 1000.0 / elapsed_ms if elapsed_ms > 0.0 else float("inf")
        samples_per_s = batches_per_s * batch_size
        speedup_vs_eager = eager_elapsed_ms / elapsed_ms if elapsed_ms > 0.0 else float("inf")
        mem_delta_mb_vs_eager = max_mem_mb - eager_mem_mb
        rows.append(
            BenchmarkResult(
                mode=mode_name,
                backend=backend,
                metric_name=metric_name,
                elapsed_ms=elapsed_ms,
                batches_per_s=batches_per_s,
                samples_per_s=samples_per_s,
                max_mem_mb=max_mem_mb,
                speedup_vs_eager=speedup_vs_eager,
                mem_delta_mb_vs_eager=mem_delta_mb_vs_eager,
            )
        )
    return rows


def print_mode_report(
    *,
    mode_name: str,
    diff: float,
    dropout_note: str | None,
    rows: list[BenchmarkResult],
) -> None:
    print(f"\n[{mode_name}]")
    print("validation: shape match + finite outputs passed")
    if dropout_note is not None:
        print(dropout_note)
    print(f"{'backend':>7s} | {'metric_ms':>12s} | {'batches/s':>10s} | {'samples/s':>10s} | {'max_mem_mb':>10s} | {'speedup':>8s} | {'mem_delta':>10s}")
    print("-" * 91)
    for row in rows:
        print(
            f"{row.backend:>7s} | "
            f"{row.elapsed_ms:12.3f} | "
            f"{row.batches_per_s:10.2f} | "
            f"{row.samples_per_s:10.2f} | "
            f"{row.max_mem_mb:10.2f} | "
            f"{row.speedup_vs_eager:8.3f} | "
            f"{row.mem_delta_mb_vs_eager:10.2f}"
        )
    print(f"max_abs_diff(eager, sdpa): {diff:.6e}")


def save_results_json(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)


def save_results_csv(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(BenchmarkResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    if device.type != "cuda":
        raise ValueError("This benchmark is intended for CUDA devices such as A100.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    q, k, v = build_qkv(args, device, dtype)
    print(
        "shape:",
        f"B={args.batch_size}, H={args.num_heads}, Q={args.query_len}, K={args.key_len}, D={args.head_dim}, dtype={args.dtype}",
    )

    all_rows: list[BenchmarkResult] = []
    for training in (True, False):
        mode_name = "train" if training else "eval"
        deterministic_dropout = 0.0 if training and args.dropout > 0.0 else args.dropout
        diff = compare_deterministic_outputs(
            q,
            k,
            v,
            training=training,
            dropout_p=deterministic_dropout,
            seed=args.seed,
        )
        validate_configured_outputs(q, k, v, training=training, dropout_p=args.dropout, seed=args.seed)
        backend_timings: dict[str, tuple[float, float]] = {}
        for backend in ("eager", "sdpa", "auto"):
            backend_timings[backend] = benchmark_backend(
                backend,
                q,
                k,
                v,
                training=training,
                dropout_p=args.dropout,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        rows = build_result_rows(mode_name=mode_name, batch_size=args.batch_size, backend_timings=backend_timings)
        all_rows.extend(rows)
        dropout_note = None
        if training and args.dropout > 0.0:
            dropout_note = "note: diff uses deterministic dropout=0.0; table timings keep configured train dropout."
        print_mode_report(mode_name=mode_name, diff=diff, dropout_note=dropout_note, rows=rows)

    if args.output_json:
        save_results_json(Path(args.output_json), all_rows)
        print(f"\nsaved_json: {args.output_json}")
    if args.output_csv:
        save_results_csv(Path(args.output_csv), all_rows)
        print(f"saved_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
