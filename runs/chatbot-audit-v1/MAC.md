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
