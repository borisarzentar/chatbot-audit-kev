# Run the trained audit model on Apple Silicon

From the cloned repository on your Mac, with uv installed:

```bash
uv sync --extra serve --python 3.13
HF_HUB_DISABLE_TELEMETRY=1 KEV_BACKEND=mlx \
uv run --extra serve python -m kev.serve \
  --run runs/chatbot-audit-local/full-v1-study/checkpoint \
  --host 127.0.0.1 --port 8009
```

The first start downloads the pinned Qwen base model from Hugging Face. The trained
adapter and pointer head load from this repository. No manual conversion is needed.
The startup message should say `via mlx`. This checkpoint has been tested on CUDA;
its Apple Silicon execution has not yet been verified and probabilities may differ slightly.

Open http://127.0.0.1:8009/docs to try the API. Or, from another terminal in the repository:

```bash
uv run python runs/chatbot-audit-v1/ask_local.py runs/chatbot-audit-v1/example-state.json
```

Replace the final path with your own session JSON. Questions are in `questions.json`.
The model was trained on synthetic ecommerce audits; its confidence is not calibrated
on real conversations. Keep the server terminal open; Ctrl+C stops it.

## Benchmark the audit workload

With the server running, use a second terminal in the repository:

```bash
uv run python -m kev.benchmark \
  --remote http://127.0.0.1:8009 \
  --data runs/chatbot-audit-v1/data/development.jsonl \
  --out runs/audit-mac-development
```

Choose a new output directory for each run. This evaluates all 324 development
sessions and 1,620 questions through the actual Mac server. It writes `rows.json`
and `report.json`. CUDA and MLX probabilities may differ; this is a Mac validation,
not a promise of identical results. Remote rows do not contain raw logits and must
not be used to refit the checkpoint temperature.

The Git exceptions include `calibration.jsonl`, `development.jsonl`, `test.jsonl`,
their annotation evidence, and the original hash manifest. Training data remains
local. The manifest describes the full original dataset, including that omitted
training partition. The locked test file is included for portability only; it has
not been evaluated. Reserve it for a final check after development decisions.

## Reproduce workload calibration

The checkpoint temperature is fitted on the 324-session calibration partition,
not on the development benchmark. Saved CUDA logits and reports are included so
the fit can be reproduced without running another GPU evaluation:

```bash
uv run python scripts/calibrate_checkpoint.py \
  --run runs/chatbot-audit-local/full-v1-study/checkpoint \
  --rows runs/chatbot-audit-local/full-v1-study/calibration-raw/rows.json \
  --transfer runs/chatbot-audit-local/full-v1-study/trained/rows.json \
  --allow-in-distribution
```

This command writes `head.pt`; restart the server afterward. Unset
`KEV_TEMPERATURE` if you previously set it, so the server uses the saved value.
`--allow-in-distribution` is intentional: these are different scenario groups
from the same synthetic source, not independent held-out datasets. The generic
script cannot resolve this custom JSONL corpus to a registered Kev suite and
records that limitation in the checkpoint. This is workload calibration only,
not evidence of calibrated confidence on real conversations or other domains.
`calibration-summary.json` records the split checks and the development results.
