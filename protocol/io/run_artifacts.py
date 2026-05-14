from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from protocol.manifests import Manifest


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git(cmd: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _resolve_baseline_root(repo_root: Path, baseline_name: str) -> Path:
    baselines_root = repo_root / "baselines"
    preferred = baselines_root / baseline_name
    if preferred.exists():
        return preferred
    for candidate in baselines_root.iterdir():
        if candidate.name.lower() == baseline_name.lower():
            return candidate
    return preferred


def build_git_meta(repo_root: Path, baseline_name: str) -> dict[str, Any]:
    baseline_root = _resolve_baseline_root(repo_root, baseline_name)
    upstream_info = baseline_root / "UPSTREAM_INFO.json"
    upstream = json.loads(upstream_info.read_text(encoding="utf-8")) if upstream_info.exists() else {}
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "main_repo": {
            "head": _git(["git", "rev-parse", "HEAD"], repo_root),
            "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "status_short": _git(["git", "status", "--short"], repo_root),
        },
        "baseline": upstream,
    }


def generate_run_id(manifest: Manifest) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{timestamp}_{manifest.baseline}_{manifest.dataset.lower()}_"
        f"pl{manifest.pred_len}_s{manifest.seed}"
    )


def write_run_snapshot(
    run_dir: Path,
    manifest: Manifest,
    command: list[str],
    env: dict[str, str],
    original_manifest: dict[str, Any] | None = None,
    batch_context: dict[str, Any] | None = None,
) -> None:
    payload = {
        "manifest_path": str(manifest.path),
        "manifest": manifest.raw,
        "original_manifest": original_manifest or manifest.raw,
        "batch_context": batch_context,
        "resolved_command": command,
        "selected_env": {
            key: value
            for key, value in env.items()
            if key in manifest.env or key in {"PYTHONPATH", "CUDA_VISIBLE_DEVICES"}
        },
    }
    write_json(run_dir / "config.snapshot.json", payload)
