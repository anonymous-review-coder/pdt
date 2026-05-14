from __future__ import annotations

from pathlib import Path
import sys

from protocol.manifests import Manifest
from protocol.runners.common import build_base_env, build_cli_args


CANONICAL_PDT_ENTRY = Path("baselines/PDT/run.py")
LEGACY_PDT_OLD_ENTRY = Path("baselines/PDT_old/run_IN.py")


def build_command(repo_root: Path, manifest: Manifest, run_dir: Path):
    args = dict(manifest.args)
    if not isinstance(manifest.pred_len, int):
        raise TypeError("PDT builder expects a single pred_len after manifest expansion.")
    if manifest.repo_relative_entry == LEGACY_PDT_OLD_ENTRY:
        protocol_only_args = {
            "metric_policy",
            "output_dir",
            "run_id",
            "seed",
            "selection_policy",
            "skip_predictions",
        }.intersection(args)
        if protocol_only_args:
            names = ", ".join(sorted(protocol_only_args))
            raise ValueError(f"Legacy PDT_old route cannot pass protocol-only args to run_IN.py: {names}")
        args["checkpoints"] = str(run_dir / "checkpoints")
        args["results"] = str(run_dir / "results")
        args["test_results"] = str(run_dir / "test_results")
        args["log_path"] = str(run_dir / "result_long_term_forecast.txt")
        command = ["python", "-u", "run_IN.py", *build_cli_args(args)]
        env = dict(manifest.env)
        cwd = repo_root / LEGACY_PDT_OLD_ENTRY.parent
        return command, env, cwd
    if manifest.repo_relative_entry != CANONICAL_PDT_ENTRY:
        raise ValueError(
            "PDT experiments must use either the protocol PDT route "
            f"{CANONICAL_PDT_ENTRY.as_posix()} or the legacy old-version route "
            f"{LEGACY_PDT_OLD_ENTRY.as_posix()}, got {manifest.repo_relative_entry.as_posix()}."
        )

    trials = manifest.search_budget.get("trials")
    if trials is not None:
        args.setdefault("itr", int(trials))

    args.setdefault("task_name", "long_term_forecast")
    args.setdefault("is_training", 1)
    args.setdefault("model", "PDT")
    args.setdefault("model_id", f"{manifest.dataset.lower()}_pl{manifest.pred_len}")
    args.setdefault("des", "protocol")
    args["pred_len"] = manifest.pred_len
    args["seed"] = manifest.seed
    args["run_id"] = run_dir.name
    args["output_dir"] = str(run_dir)
    args["metric_policy"] = manifest.metric_policy
    args["selection_policy"] = manifest.selection_policy

    entry = repo_root / CANONICAL_PDT_ENTRY
    command = [sys.executable, "-u", str(entry), *build_cli_args(args)]
    env = build_base_env(repo_root, manifest)
    return command, env
