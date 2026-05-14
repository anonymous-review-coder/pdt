from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import resource
import sys
from typing import Any

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "PDT"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models.PDT import Model
from protocol.manifests import load_manifest


DEFAULT_ARGS: dict[str, Any] = {
    "activation": "gelu",
    "alpha_init": 0.5,
    "CKA_flag": 0,
    "dropout": 0.1,
    "embed_size": 8,
    "freeze_R": 0,
    "mask_sharpness_k": 100.0,
    "mask_threshold": 0.25,
    "Q_chan_indep": 0,
    "temp_patch_len": 16,
    "temp_stride": 8,
}


def _select_pred_len_args(raw: dict[str, Any], pred_len: int) -> dict[str, Any]:
    args = dict(raw["args"])
    manifest_pred_len = raw["pred_len"]
    if isinstance(manifest_pred_len, list):
        if pred_len not in manifest_pred_len:
            raise ValueError(f"pred_len={pred_len} is not in manifest pred_len list: {manifest_pred_len}")
        index = manifest_pred_len.index(pred_len)
        for key, value in list(args.items()):
            if isinstance(value, list):
                args[key] = value[index]
    args["pred_len"] = pred_len
    return args


def _resolve_baseline_path(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str((BASELINE_ROOT / path).resolve())


def _build_model_args(manifest_path: str, pred_len: int) -> Namespace:
    manifest = load_manifest(manifest_path)
    args = DEFAULT_ARGS | _select_pred_len_args(manifest.raw, pred_len)
    args["root_path"] = str((BASELINE_ROOT / "dataset").resolve())
    for field in (
        "q_mat_file",
        "Q_MAT_file",
        "q_out_mat_file",
        "Q_OUT_MAT_file",
        "r_mat_file",
        "R_MAT_file",
        "rk_mat_file",
        "Rk_MAT_file",
    ):
        args[field] = _resolve_baseline_path(args.get(field))
    return Namespace(**args)


def _estimate_module_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if not inputs:
            return
        in_features = module.in_features
        output_elements = output.numel()
        macs += int(output_elements * in_features)

    def conv1d_hook(module: nn.Conv1d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
        macs += int(output.numel() * kernel_ops)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))

    with torch.no_grad():
        model(sample)
    for hook in hooks:
        hook.remove()
    return macs


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    model_args = _build_model_args(args.manifest, args.pred_len)
    device = torch.device(args.device)
    model = Model(model_args).to(device).eval()
    sample = torch.randn(args.batch_size, model_args.seq_len, model_args.enc_in, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    macs = _estimate_module_macs(model, sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    else:
        after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_bytes = max(after_rss - before_rss, 0)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "manifest": str(Path(args.manifest).resolve()),
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "device": str(device),
        "parameter_count": int(parameter_count),
        "estimated_macs": int(macs),
        "estimated_macs_g": macs / 1e9,
        "peak_memory_bytes": int(peak_memory_bytes),
        "peak_memory_mb": peak_memory_bytes / (1024 ** 2),
        "note": "MACs are estimated from Linear and Conv1d module hooks for one forward pass.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile PDT MACs and memory for one manifest setting.")
    parser.add_argument("--manifest", required=True, help="Path to a PDT manifest JSON file.")
    parser.add_argument("--pred-len", type=int, required=True, help="Prediction horizon to profile.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic profiling batch size.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Profiling device.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested, but CUDA is not available.")

    result = _profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
