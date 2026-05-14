# PDT Anonymous Review Code

This repository is an anonymized review package for PDT. It contains only the
minimal code needed to train, evaluate, and profile the PDT model used in the
revised manuscript.

## Contents

- `baselines/PDT/`: PDT model, data loaders, training loop, and evaluation loop.
- `protocol/`: manifest parser and PDT runner.
- `experiments/stage1/pdt/*.json`: parameter manifests for the reported PDT runs.
- `scripts/local/run_manifest.py`: local manifest entrypoint.
- `scripts/remote/run_manifest.sh`: remote-friendly manifest wrapper.
- `scripts/measure_pdt_weather_resources.py`: thop-based resource measurement
  for PDT on Weather with `L=96,T=192`.
- `baselines/PDT/dataset/RRR_mats/`: PDT initialization matrices referenced by
  the manifests.
- `baselines/PDT/dataset/ETT-small/ETTh1.csv`: small public dataset file used
  for the included smoke check.

Large benchmark datasets are not bundled. Download the public ETT, Electricity,
Traffic, Weather, and Exchange datasets and place them under
`baselines/PDT/dataset/` using the paths referenced by the manifests.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The revised manuscript reports experiments with PyTorch 2.5.1 on an NVIDIA RTX
3090 GPU. The code also supports CPU execution for smoke checks and profiling.

## Dry Run

```bash
python scripts/local/run_manifest.py \
  --manifest experiments/stage1/pdt/etth1_smoke.json \
  --run-id dryrun_etth1 \
  --output-root /tmp/pdt_review_dryrun \
  --dry-run
```

## Smoke Training And Evaluation

The following command runs a short ETTh1 train/evaluate check using the included
ETTh1 data and PDT matrices. It is intended to verify code execution, not to
reproduce the manuscript metric values.

```bash
python scripts/local/run_manifest.py \
  --manifest experiments/stage1/pdt/etth1_smoke.json \
  --run-id smoke_etth1_pl96 \
  --output-root /tmp/pdt_review_runs \
  --set args.train_epochs=1 \
  --set args.patience=1 \
  --set args.batch_size=64 \
  --set args.num_workers=0
```

Full reproduction uses the corresponding multi-horizon manifests, for example:

```bash
python scripts/local/run_manifest.py \
  --manifest experiments/stage1/pdt/etth1_multi_pred_len.json \
  --output-root artifacts/runs
```

## MACs And Memory Profiling

For the reviewer-facing Weather `L=96,T=192` setting, use the thop-based script
adapted from the original resource-measurement code:

```bash
python scripts/measure_pdt_weather_resources.py --device cpu
```

On a CUDA machine, run:

```bash
python scripts/measure_pdt_weather_resources.py --device cuda
```

The script reports MACs, parameter count, inference memory, and training memory for PDT. 

```bash
python scripts/profile_macs_memory.py \
  --manifest experiments/stage1/pdt/etth1_smoke.json \
  --pred-len 96 \
  --batch-size 1 \
  --device cpu
```

When CUDA is available, use `--device cuda` to report peak CUDA memory. MACs are
estimated with forward hooks over the Linear and Conv1d modules used by the PDT
forecasting path.
