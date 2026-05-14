from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import resource
import sys
from typing import Any

import torch

try:
    from thop import profile
except ImportError as exc:
    raise SystemExit("Missing dependency: thop. Install it with `pip install thop`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "PDT"
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "stage1" / "pdt" / "weather_multi_pred_len.json"
FIXED_PRED_LEN = 192
MODEL_NAME = "PDT"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models import PDT
from protocol.manifests import load_manifest


DEFAULT_ARGS: dict[str, Any] = {
    "activation": "gelu",
    "CKA_flag": 0,
    "distil": True,
    "embed": "timeF",
    "factor": 3,
    "freeze_R": 0,
    "label_len": 48,
    "mask_sharpness_k": 100.0,
    "output_attention": False,
    "Q_chan_indep": 0,
    "temp_patch_len": 16,
    "temp_stride": 8,
}


def _select_pred_len_args(manifest_path: Path, pred_len: int) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if manifest.dataset != "Weather":
        raise ValueError(f"This resource script only supports Weather, got {manifest.dataset}.")
    if manifest.seq_len != 96:
        raise ValueError(f"This resource script only supports seq_len=96, got {manifest.seq_len}.")
    if pred_len not in manifest.pred_lens:
        raise ValueError(f"pred_len={pred_len} is not listed in {manifest_path}.")

    args = dict(manifest.raw["args"])
    index = manifest.pred_lens.index(pred_len)
    for key, value in list(args.items()):
        if isinstance(value, list):
            args[key] = value[index]
    args["pred_len"] = pred_len
    args["model"] = MODEL_NAME
    return args


def _resolve_baseline_path(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str((BASELINE_ROOT / path).resolve())


def build_config(manifest_path: Path) -> Namespace:
    args = DEFAULT_ARGS | _select_pred_len_args(manifest_path, FIXED_PRED_LEN)
    args["root_path"] = str((BASELINE_ROOT / "dataset" / "weather").resolve())
    for field in ("q_mat_file", "Q_MAT_file", "r_mat_file", "R_MAT_file", "rk_mat_file", "Rk_MAT_file"):
        args[field] = _resolve_baseline_path(args.get(field))
    return Namespace(**args)


def _memory_snapshot() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def measure_inference(model: torch.nn.Module, config: Namespace, device: torch.device) -> dict[str, float | int | None]:
    model.eval()
    x_enc = torch.randn(1, config.seq_len, config.enc_in, device=device)
    x_mark_enc = torch.randn(1, config.seq_len, 4, device=device)
    x_dec = torch.randn(1, config.pred_len + config.label_len, config.dec_in, device=device)
    x_mark_dec = torch.randn(1, config.pred_len + config.label_len, 4, device=device)
    inputs = (x_enc, x_mark_enc, x_dec, x_mark_dec)

    macs, params = profile(model, inputs=inputs, verbose=False)

    cuda_peak_memory = None
    cpu_rss_delta = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize(device)
        cuda_peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        before = _memory_snapshot()
        with torch.no_grad():
            _ = model(*inputs)
        after = _memory_snapshot()
        cpu_rss_delta = max(after - before, 0)

    return {
        "macs": int(macs),
        "macs_g": float(macs / 1e9),
        "params": int(params),
        "params_m": float(params / 1e6),
        "inference_cuda_peak_memory_mb": None if cuda_peak_memory is None else cuda_peak_memory / (1024**2),
        "inference_cpu_rss_delta_kb": cpu_rss_delta,
    }


def measure_training_memory(config: Namespace, device: torch.device, batch_size: int) -> dict[str, float | int | None]:
    model = PDT.Model(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in, device=device)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4, device=device)
    x_dec = torch.randn(batch_size, config.pred_len + config.label_len, config.dec_in, device=device)
    x_mark_dec = torch.randn(batch_size, config.pred_len + config.label_len, 4, device=device)
    target = torch.randn(batch_size, config.pred_len, config.c_out, device=device)

    cuda_peak_memory = None
    cpu_rss_delta = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    before = _memory_snapshot()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        after = _memory_snapshot()
        cpu_rss_delta = max(after - before, 0)

    return {
        "training_batch_size": batch_size,
        "training_cuda_peak_memory_mb": None if cuda_peak_memory is None else cuda_peak_memory / (1024**2),
        "training_cpu_rss_delta_kb": cpu_rss_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure PDT resources for Weather L=96, T=192.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Weather PDT manifest path.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Batch size for training-memory measurement.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    config = build_config(args.manifest.resolve())
    device = torch.device(args.device)
    model = PDT.Model(config).to(device)

    result: dict[str, Any] = {
        "dataset": "Weather",
        "model": MODEL_NAME,
        "seq_len": config.seq_len,
        "pred_len": FIXED_PRED_LEN,
        "device": str(device),
        "manifest": str(args.manifest.resolve()),
    }
    result.update(measure_inference(model, config, device))
    result.update(measure_training_memory(config, device, args.train_batch_size))

    print(
        f"{'Dataset':<10} | {'L':<4} | {'T':<4} | {'Model':<8} | "
        f"{'MACs (G)':<10} | {'Params (M)':<10} | {'Inf Mem (MB)':<12} | {'Train Mem (MB)':<14}"
    )
    print("-" * 96)
    inf_mem = result["inference_cuda_peak_memory_mb"]
    train_mem = result["training_cuda_peak_memory_mb"]
    print(
        f"{result['dataset']:<10} | {result['seq_len']:<4} | {result['pred_len']:<4} | {result['model']:<8} | "
        f"{result['macs_g']:<10.4f} | {result['params_m']:<10.4f} | "
        f"{(inf_mem if inf_mem is not None else 0):<12.2f} | {(train_mem if train_mem is not None else 0):<14.2f}"
    )
    if device.type != "cuda":
        print("CUDA is not available; CUDA memory fields are 0 and CPU RSS deltas are reported in JSON.")

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
