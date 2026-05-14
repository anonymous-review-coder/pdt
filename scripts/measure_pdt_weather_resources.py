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
FIXED_PRED_LEN = 192
MODEL_NAME = "PDT"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models import PDT


OLD_R2LINEAR_WEATHER_ARGS: dict[str, Any] = {
    "activation": "gelu",
    "CKA_flag": 0,
    "channel_independence": 0,
    "class_strategy": "projection",
    "c_out": 21,
    "data": "custom",
    "data_path": "weather.csv",
    "d_ff": 2048,
    "d_layers": 1,
    "dec_in": 21,
    "decomposition": 0,
    "d_model": 512,
    "distil": True,
    "embed": "timeF",
    "embed_size": 16,
    "enc_in": 21,
    "e_layers": 2,
    "factor": 1,
    "features": "M",
    "fc_dropout": 0.05,
    "freq": "h",
    "freeze_R": 0,
    "head_dropout": 0.0,
    "individual": False,
    "is_training": 0,
    "kernel_size": 25,
    "k_top": 16,
    "label_len": 48,
    "mask_sharpness_k": 2,
    "mask_threshold": 0.1,
    "model": MODEL_NAME,
    "model_id": "test",
    "moving_avg": 25,
    "n_heads": 8,
    "num_kernels": 6,
    "output_attention": False,
    "padding_patch": "end",
    "patch_len": 16,
    "pred_len": FIXED_PRED_LEN,
    "Q_chan_indep": 0,
    "revin": 1,
    "r_rank": 96,
    "seg_len": 48,
    "seq_len": 96,
    "stride": 8,
    "subtract_last": 0,
    "target": "OT",
    "task_name": "long_term_forecast",
    "temp_patch_len": 16,
    "temp_stride": 8,
    # "top_k": 5,
    "use_norm": 1,
    "win_size": 1,
    "affine": 0,
    "alpha_init": 0.4,
    "cross_activation": "tanh",
    "dropout": 0.0,
}


def build_config() -> Namespace:
    args = dict(OLD_R2LINEAR_WEATHER_ARGS)
    args["root_path"] = str((BASELINE_ROOT / "dataset" / "weather").resolve())
    rrr_root = BASELINE_ROOT / "dataset" / "RRR_mats" / "weather"
    args["q_mat_file"] = str((rrr_root / "weather_RRR_L96_R96_H192_Qin.npy").resolve())
    args["Q_MAT_file"] = args["q_mat_file"]
    args["r_mat_file"] = str((rrr_root / "weather_RRR_L96_R96_H192_R.npy").resolve())
    args["R_MAT_file"] = args["r_mat_file"]
    args["rk_mat_file"] = str((rrr_root / "weather_RRR_L96_R16_H192_R.npy").resolve())
    args["Rk_MAT_file"] = args["rk_mat_file"]
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Batch size for training-memory measurement.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    config = build_config()
    device = torch.device(args.device)
    model = PDT.Model(config).to(device)

    result: dict[str, Any] = {
        "dataset": "Weather",
        "model": MODEL_NAME,
        "seq_len": config.seq_len,
        "pred_len": FIXED_PRED_LEN,
        "device": str(device),
        "config_source": "old/measure_resources_baselines.py R2Linear weather pl=192 with model=PDT",
        "d_ff": config.d_ff,
        "k_top": config.k_top,
        "mask_sharpness_k": config.mask_sharpness_k,
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
