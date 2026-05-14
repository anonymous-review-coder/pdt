from __future__ import annotations

from pathlib import Path
from typing import Any

from protocol.manifests import Manifest


def build_cli_args(args: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if value is None:
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def build_base_env(repo_root: Path, manifest: Manifest) -> dict[str, str]:
    env = dict(manifest.env)
    pythonpath_parts = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env
