from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from protocol.io import build_git_meta, ensure_dir, generate_run_id, write_json, write_run_snapshot
from protocol.manifests import load_manifest, manifest_from_data
from protocol.runners.pdt import build_command as build_pdt_command


BUILDERS = {
    "pdt": build_pdt_command,
}


def _parse_override_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _apply_override(data: dict, dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = _parse_override_value(raw_value)


def _apply_overrides(manifest, overrides: list[str]):
    if not overrides:
        return manifest
    updated = copy.deepcopy(manifest.raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        _apply_override(updated, key, value)
    return manifest_from_data(updated, manifest.path)


def _expand_batch_manifests(manifest) -> list[tuple[Any, dict[str, Any] | None]]:
    pred_lens = manifest.pred_lens
    if isinstance(manifest.pred_len, int):
        return [(manifest, None)]

    original_manifest = copy.deepcopy(manifest.raw)
    args = original_manifest.get("args", {})
    if not isinstance(args, dict):
        raise TypeError("Manifest field 'args' must be an object.")

    aligned_args: dict[str, list[Any]] = {}
    broadcast_args: dict[str, Any] = {}
    for key, value in args.items():
        if key == "pred_len":
            continue
        if isinstance(value, list):
            if any(isinstance(item, (list, dict)) for item in value):
                raise ValueError(f"Manifest args.{key} cannot be a nested list/dict for pred_len alignment.")
            if len(value) != len(pred_lens):
                raise ValueError(
                    f"Manifest args.{key} length {len(value)} does not match pred_len length {len(pred_lens)}."
                )
            aligned_args[key] = value
        else:
            broadcast_args[key] = value

    expanded: list[tuple[Any, dict[str, Any] | None]] = []
    total = len(pred_lens)
    for index, active_pred_len in enumerate(pred_lens):
        child_raw = copy.deepcopy(original_manifest)
        child_raw["pred_len"] = active_pred_len
        child_args = dict(broadcast_args)
        for key, values in aligned_args.items():
            child_args[key] = values[index]
        child_args["pred_len"] = active_pred_len
        child_raw["args"] = child_args
        child_manifest = manifest_from_data(child_raw, manifest.path)
        batch_context = {
            "index": index,
            "size": total,
            "active_pred_len": active_pred_len,
            "batch_pred_lens": pred_lens,
        }
        expanded.append((child_manifest, batch_context))
    return expanded


def _resolve_run_id(manifest, cli_run_id: str | None, batch_size: int, active_pred_len: int) -> str:
    if cli_run_id is None:
        return generate_run_id(manifest)
    if batch_size == 1:
        return cli_run_id
    return f"{cli_run_id}_pl{active_pred_len}"


def _stream_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def _setting_from_args(args: dict[str, Any], itr_index: int = 0) -> str:
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_rr{}_frR{}_k{}_{}_{}".format(
        args.get("task_name", "long_term_forecast"),
        args["model_id"],
        args["model"],
        args["data"],
        args["features"],
        args["seq_len"],
        args["label_len"],
        args["pred_len"],
        args["d_model"],
        args.get("n_heads", 8),
        args["e_layers"],
        args["d_layers"],
        args["d_ff"],
        args["factor"],
        args.get("embed", "timeF"),
        args.get("distil", True),
        args["r_rank"],
        args["freeze_R"],
        args["k_top"],
        args["des"],
        itr_index,
    )


def _resolve_legacy_output_path(cwd: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else cwd / path


def _legacy_artifact_path(root: Path, setting: str, filename: str) -> Path:
    exact = root / setting / filename
    if exact.exists():
        return exact
    if setting.endswith("_0"):
        without_itr = root / setting[:-2] / filename
        if without_itr.exists():
            return without_itr
    return exact


def _postprocess_legacy_pdt_old(manifest, run_dir: Path, cwd: Path) -> None:
    if manifest.repo_relative_entry.as_posix() != "baselines/PDT_old/run_IN.py":
        return

    import numpy as np

    args = manifest.args
    setting = _setting_from_args(args)
    checkpoint_root = str(args.get("checkpoints", run_dir / "checkpoints"))
    results_root = str(args.get("results", run_dir / "results"))
    checkpoint_base = _resolve_legacy_output_path(cwd, checkpoint_root)
    results_base = _resolve_legacy_output_path(cwd, results_root)
    checkpoint = _legacy_artifact_path(checkpoint_base, setting, "checkpoint.pth")
    metrics_npy = _legacy_artifact_path(results_base, setting, "metrics.npy")

    if not checkpoint.exists():
        matches = sorted(checkpoint.parent.parent.glob("*/checkpoint.pth")) if checkpoint.parent.parent.exists() else []
        if len(matches) == 1:
            checkpoint = matches[0]
        else:
            candidates = "\n".join(str(path) for path in matches[:20])
            raise FileNotFoundError(
                f"Legacy PDT checkpoint not found after training: {checkpoint}"
                + (f"\nCandidate checkpoints:\n{candidates}" if candidates else "")
            )
    if not metrics_npy.exists():
        matches = sorted(metrics_npy.parent.parent.glob("*/metrics.npy")) if metrics_npy.parent.parent.exists() else []
        if len(matches) == 1:
            metrics_npy = matches[0]
        else:
            candidates = "\n".join(str(path) for path in matches[:20])
            raise FileNotFoundError(
                f"Legacy PDT metrics.npy not found after training: {metrics_npy}"
                + (f"\nCandidate metrics:\n{candidates}" if candidates else "")
            )

    shutil.copy2(checkpoint, run_dir / "best.ckpt")
    shutil.copy2(metrics_npy, run_dir / "metrics.npy")

    values = np.load(metrics_npy).astype(float).tolist()
    if len(values) < 5:
        raise ValueError(f"Legacy PDT metrics.npy should contain at least 5 values: {metrics_npy}")
    metrics = {
        "mae": values[0],
        "mse": values[1],
        "rmse": values[2],
        "mape": values[3],
        "mspe": values[4],
    }
    write_json(run_dir / "metrics.json", metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a baseline experiment from a protocol manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest.")
    parser.add_argument("--run-id", help="Override the generated run id.")
    parser.add_argument("--output-root", help="Root directory for run artifacts.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override manifest values, e.g. --set pred_len=192 --set args.pred_len=192",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing.")
    cli_args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(cli_args.manifest)
    manifest = _apply_overrides(manifest, cli_args.overrides)
    builder = BUILDERS.get(manifest.baseline.lower())
    if builder is None:
        raise ValueError(f"Unsupported baseline: {manifest.baseline}")

    output_root = Path(cli_args.output_root).resolve() if cli_args.output_root else repo_root / "artifacts" / "runs"
    original_manifest = copy.deepcopy(manifest.raw)
    expanded_manifests = _expand_batch_manifests(manifest)

    for child_manifest, batch_context in expanded_manifests:
        if not isinstance(child_manifest.pred_len, int):
            raise TypeError("Expanded child manifest must have a single pred_len.")
        run_id = _resolve_run_id(
            child_manifest,
            cli_args.run_id,
            len(expanded_manifests),
            child_manifest.pred_len,
        )
        run_dir = ensure_dir(output_root / run_id)
        build_result = builder(repo_root, child_manifest, run_dir)
        if len(build_result) == 2:
            command, runner_env = build_result
            run_cwd = repo_root
        elif len(build_result) == 3:
            command, runner_env, run_cwd = build_result
        else:
            raise ValueError(f"Invalid builder result for baseline {child_manifest.baseline}: {build_result!r}")
        write_run_snapshot(
            run_dir,
            child_manifest,
            command,
            runner_env,
            original_manifest=original_manifest,
            batch_context=batch_context,
        )
        write_json(run_dir / "git_meta.json", build_git_meta(repo_root, child_manifest.baseline.lower()))

        if cli_args.dry_run:
            print("Resolved run directory:")
            print(run_dir)
            print("Resolved working directory:")
            print(run_cwd)
            print("Resolved command:")
            print(" ".join(command))
            continue

        return_code = _stream_process(command, run_cwd, runner_env, run_dir / "train.log")
        if return_code != 0:
            return return_code
        _postprocess_legacy_pdt_old(child_manifest, run_dir, run_cwd)

        print(f"Run finished successfully: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
