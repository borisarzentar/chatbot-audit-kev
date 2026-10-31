# Kev

Small Jev-like decision models you can train and run yourself.

<p>
  <a href="https://github.com/jaredpalmer/kev/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/jaredpalmer/kev/ci.yml?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd"><img alt="Weights: Kev-0.8B · 4B · 9B · 27B" src="https://img.shields.io/badge/WEIGHTS-0.8B%20%C2%B7%204B%20%C2%B7%209B%20%C2%B7%2027B-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/spaces/jaredpalmer/kev"><img alt="Demo on Hugging Face Spaces" src="https://img.shields.io/badge/DEMO-HF%20Spaces-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/datasets/jaredpalmer/kev-suites"><img alt="Frozen eval suites" src="https://img.shields.io/badge/EVAL%20SUITES-frozen-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

Kev is a family of small decision models built on Qwen3.5 and Qwen3.8 and based on the architecture described in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked). You can use the pretrained weights or train your own. The API matches TypeSafe's [System One](https://docs.typesafe.ai/api), so you can point their Python SDK at your local server.

## Chatbot Audit Model

This fork adds a Kev model that audits customer-support chatbot sessions. You give it one session (the customer's messages, the agent's replies and every tool call with its result, as JSON) and it answers five questions about it:

| Question | Type | Asks |
|---|---|---|
| `issue_resolved` | `noul` | Was the customer's issue fully resolved within the session? |
| `factually_consistent` | `noul` | Does everything the agent said match its own tool results? |
| `policy_adherence` | `noul` | Did the agent follow the policies its own tool calls returned? |
| `escalation_needed` | `noul` | Does a person need to follow up? |
| `customer_sentiment` | `choice` | Is the customer positive, neutral, negative or mixed at the end? |

The exact wording is in [`questions.json`](runs/chatbot-audit-v1/questions.json) and an example session is in [`example-state.json`](runs/chatbot-audit-v1/example-state.json).

The model is Kev-0.8B (`jaredpalmer/kev-0.8b@bf75a6a8`) fine-tuned for one epoch on 1,944 synthetic ecommerce sessions (LoRA rank 16, lr 2e-5, states up to 4,096 tokens; 85 minutes on an RTX 3060 Ti, 4.1 GB peak GPU memory). The temperature, 2.24, was then fitted on a separate calibration split. The checkpoint is in [`runs/chatbot-audit-local/full-v1-study/checkpoint`](runs/chatbot-audit-local/full-v1-study/checkpoint): a 43 MB adapter plus the pointer head. The base model, Qwen3.5-0.8B-Base, downloads from Hugging Face on first use.

### Run the Audit Model

On a Mac (MLX) or a machine with an NVIDIA GPU:

```bash
uv sync --extra serve --python 3.13
uv run --extra serve python -m kev.serve \
  --run runs/chatbot-audit-local/full-v1-study/checkpoint --port 8009
```

On Apple Silicon the startup message should say `via mlx`. Then, from a second terminal, audit a session:

```bash
uv run python runs/chatbot-audit-v1/ask_local.py runs/chatbot-audit-v1/example-state.json
```

Replace the path with your own session JSON. `ask_local.py` sends it with the five questions to `/v1/systemone` and prints the answers, so the TypeSafe SDK and the `curl` examples in [Quick Start](#quick-start) work the same way. [`MAC.md`](runs/chatbot-audit-v1/MAC.md) has more detail.

### Audit Results

All rows are the development split: 324 sessions, 1,620 questions, from scenario groups the model never trained on. ECE measures how far the confidence is from the real accuracy; Brier scores the whole probability distribution. Lower is better for both. "Automated at 5% error" is the share of answers you can accept automatically, most confident first, while keeping the error rate under 5%.

Each result lives in `runs/audit-<hardware>-<split>-<model>[-server]/`: `mac-m2pro-16gb` is a MacBook Pro with an M2 Pro and 16 GB of memory (MLX, bf16), `pc-rtx3060ti` a PC with an NVIDIA RTX 3060 Ti (8 GB, CUDA, bf16) and `jev` TypeSafe's hosted API. The split is `development`, `calibration` or `decision-v7`. The model is `calibrated` (the shipped audit model), `uncalibrated` (the same weights at temperature 1) or `parent` (Kev-0.8B before fine-tuning). `-server` marks a run scored through `kev.serve` over HTTP instead of in-process; those rows carry no logits.

> [!NOTE]
> The checkpoint's `head.pt` records that its temperature was fitted on `runs/chatbot-audit-local/full-v1-study/calibration-raw/rows.json`. That file was later renamed, unchanged, to [`runs/audit-pc-rtx3060ti-calibration-uncalibrated/rows.json`](runs/audit-pc-rtx3060ti-calibration-uncalibrated/rows.json). The record in `head.pt` keeps the original path on purpose.

| System | Accuracy | ECE | Brier | Automated at 5% error | Report |
|---|---|---|---|---|---|
| Kev-0.8B before fine-tuning (M2 Pro) | 0.591 | 0.191 | 0.528 | 0% | [report](runs/audit-mac-m2pro-16gb-development-parent/report.json) |
| Jev (TypeSafe API) | 0.909 | 0.064 | 0.138 | 90% | [report](runs/audit-jev-development/report.json) |
| Audit model, uncalibrated (M2 Pro) | 0.973 | 0.022 | 0.047 | 100% | [report](runs/audit-mac-m2pro-16gb-development-uncalibrated/report.json) |
| **Audit model (M2 Pro, MLX)** | **0.973** | **0.004** | **0.041** | **100%** | [report](runs/audit-mac-m2pro-16gb-development-calibrated/report.json) |
| Audit model (RTX 3060 Ti, CUDA) | 0.973 | 0.005 | 0.041 | 100% | [report](runs/audit-pc-rtx3060ti-development-calibrated/report.json) |

The M2 Pro and RTX 3060 Ti runs differ by one correct answer in 1,620, inside the noise of a paired comparison. Median latency is 0.61 s per session on the M2 Pro and 0.43 s on the RTX 3060 Ti. The paired comparisons behind these statements are written by `kev.compare` as `runs/chatbot-audit-v1/compare-<candidate>-vs-<reference>.json` (local only, not in git).

Know the limits before you rely on it:

- Every session is synthetic, generated from one ecommerce template family. These numbers say the model learned this workload, not that it will be as accurate or as well calibrated on your real conversations. Label a few hundred real sessions and check before you set thresholds.
- The temperature was fitted on different scenario groups from the same generator, not on independent data (`--allow-in-distribution`; see [`calibration-summary.json`](runs/chatbot-audit-local/full-v1-study/calibration-summary.json)).
- It is a specialist. On Kev's general `decision-v7` development set (1,264 questions, M2 Pro) accuracy fell from 0.827 to 0.820 and ECE rose from 0.034 to 0.109 ([audit model](runs/audit-mac-m2pro-16gb-decision-v7-calibrated-server/report.json), [Kev-0.8B](runs/audit-mac-m2pro-16gb-decision-v7-parent-server/report.json)). Use the released Kev models for other questions.
- The locked test split (`data/test.jsonl`, 648 sessions) has not been read. Keep it for a final check after you stop making changes.

To reproduce a row, run the benchmark in-process (the first command) or against a running server (the second). Name the output after your own hardware, so results from different machines never mix:

```bash
KEV_BACKEND=mlx uv run --extra serve python -m kev.benchmark \
  --run runs/chatbot-audit-local/full-v1-study/checkpoint \
  --data runs/chatbot-audit-v1/data/development.jsonl --out runs/audit-mac-m2pro-16gb-development-calibrated

uv run python -m kev.benchmark --remote http://127.0.0.1:8009 \
  --data runs/chatbot-audit-v1/data/development.jsonl --out runs/audit-mac-m2pro-16gb-development-calibrated-server
```

Leave out `KEV_BACKEND=mlx` on a CUDA machine. `kev.benchmark` refuses an output directory that already exists, so move a committed result aside before you re-run it.

### Fine-Tune the Audit Model Further

The 1,944-session training split is not in this repository; the [manifest](runs/chatbot-audit-v1/data/manifest.json) records its hash and label counts. To continue training, write your own sessions in the format of [`data/development.jsonl`](runs/chatbot-audit-v1/data/development.jsonl): one JSON object per line with the session as `state` and the five questions, each with a `label` (`true`/`false` for `noul`, the option name for `customer_sentiment`). Real, labelled sessions from your own chatbot are the most useful thing you can add.

Then start from this checkpoint with `--init_from`, so the run keeps what the audit model already learned. The base and revision must match the checkpoint, or the trainer refuses to load it:

```bash
uv run python -m kev.train \
  --data my-sessions.jsonl \
  --init_from runs/chatbot-audit-local/full-v1-study/checkpoint \
  --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --epochs 1 --lr 2e-5 --batch 1 --accum 2 --max_state 4096 \
  --dtype bf16 --checkpointing 1 --device cuda \
  --out runs/chatbot-audit-v2/checkpoint
```

These are the settings the current model was trained with; they fit in 4 GB of GPU memory. Train on CUDA: Apple GPUs have no fast kernels for this model's DeltaNet layers, so training on a Mac is very slow. On Windows, prefix the module with the local launcher, `uv run python runs/chatbot-audit-v1/local_runner.py kev.train ...`, which works around Unix-only imports (it also runs `kev.benchmark` and `calibrate_checkpoint`). To limit the drop on general questions, mix in Kev's own training data with `--suite evals/v7/decision-v7 --replay 2000`.

A new checkpoint needs a new temperature. Score the calibration split raw, fit on it, then compare the development split against the table above. The result directories follow the same naming, with the new model's name in front of the hardware:

```bash
KEV_TEMPERATURE=1.0 uv run python -m kev.benchmark --run runs/chatbot-audit-v2/checkpoint \
  --data runs/chatbot-audit-v1/data/calibration.jsonl --out runs/audit-v2-pc-rtx3060ti-calibration-uncalibrated

uv run python scripts/calibrate_checkpoint.py --run runs/chatbot-audit-v2/checkpoint \
  --rows runs/audit-v2-pc-rtx3060ti-calibration-uncalibrated/rows.json --allow-in-distribution

uv run python -m kev.benchmark --run runs/chatbot-audit-v2/checkpoint \
  --data runs/chatbot-audit-v1/data/development.jsonl --out runs/audit-v2-pc-rtx3060ti-development-calibrated

uv run python -m kev.compare --candidate runs/audit-v2-pc-rtx3060ti-development-calibrated \
  --reference runs/audit-pc-rtx3060ti-development-calibrated \
  --out runs/chatbot-audit-v1/compare-v2-pc-rtx3060ti-development-calibrated-vs-pc-rtx3060ti-development-calibrated.json
```

`calibrate_checkpoint.py` writes the temperature into the checkpoint's `head.pt`; restart any running server afterwards. Fit on raw logits from a local benchmark: rows from `--remote` have no logits. If your sessions come from a different source than the synthetic calibration split, hold out part of your own data and fit on that instead, since a temperature is only as good as the data it was fitted on.

The rest of this README describes the upstream Kev family the audit model is built on.

## Highlights

- Yes/no (`noul`), multiple-choice (`choice`) and rating (`score`) questions in one request. The questions share the text but can't read each other.
- Calibrated probabilities by default: each checkpoint ships with a fitted temperature.
- Drop-in for Jev: the TypeSafe Python SDK works against a Kev server unchanged.
- Four sizes, versioned together as Kev 1.0: from a 0.8B that runs on a laptop to a 27B for a single data-centre GPU.
- Documents of up to 65,536 tokens, on CUDA and on Apple Silicon through MLX. Each model card says how long a document can get before accuracy drops.
- Fine-tune on your own labelled examples. A coding-agent skill runs the whole loop on Modal, from finding your questions to serving the result.
- Deploy your own HTTPS endpoint with one command. It scales to zero when idle.
- Try it in the browser first: [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev).

## Models

Start with Kev-4B. Move to Kev-9B if you have a bigger GPU, or to Kev-27B if you have an 80 GB GPU and want the most accurate Kev. Use Kev-0.8B when size matters more than accuracy.

| Model | Base (license) | Runs on: CUDA | Runs on: Mac (MLX) | Validated context | Held-out datasets: index | Card |
|---|---|---|---|---|---|---|
| [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) | Qwen3.5-0.8B-Base (Apache-2.0) | L4, any 4 GB GPU | Any Apple Silicon Mac; measured to 65k tokens | 8,192 | 23.3 | [Details](docs/model-cards/kev-0.8b.md) |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Qwen3.5-4B-Base (Apache-2.0) | L40S, H100 | 32 GB Mac; measured to 65k tokens | 8,192 | 38.0 | [Details](docs/model-cards/kev-4b.md) |
| [Kev-9B](https://huggingface.co/jaredpalmer/kev-9b) | Qwen3.5-9B-Base (Apache-2.0) | L40S, H100 | 32 GB Mac or larger (expected, not measured) | 8,192 | 41.0 | [Details](docs/model-cards/kev-9b.md) |
| [Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) | Qwen3.8-27B, post-trained (Apache-2.0) | B200, H200, H100 80 GB | 96–128 GB Mac (expected, not measured) | 65,536 | **52.3** | [Details](docs/model-cards/kev-27b.md) |
| Jev | Hosted | TypeSafe's API | – | – | 54.0 | – |

"Held-out datasets" is the chance-corrected index of the community Decision Index, scored on the test split of `breadth-v1`: 14 public datasets in five areas that no Kev trained on. "Validated context" is the longest document, in tokens, for which accuracy on real contracts (CUAD) stays within 3 points of the same model's accuracy at 8k tokens, at the 95 % lower bound; each model card has the measurement by length.

| Model | Accuracy: New Sources | Accuracy: Trained Sources | Brier: New Sources |
|---|---|---|---|
| Kev-0.8B | 0.648 / 0.697 | 0.827 / 0.838 | 0.481 / 0.416 |
| Kev-4B | 0.817 / 0.838 | 0.873 / 0.865 | 0.269 / 0.242 |
| Kev-9B | 0.820 / 0.852 | 0.874 / 0.873 | 0.289 / 0.217 |
| Kev-27B | **0.851 / 0.889** | 0.865 / 0.866 | **0.225 / 0.156** |
| Jev | 0.857 / – | 0.845 / – | 0.211 / – |

Each cell is **development / test**. "New sources" means datasets and policy rules Kev never saw during training. It is the closest thing here to your own questions. "Trained sources" means held-out examples from the datasets Kev was trained on. We pick checkpoints using the development sets and read each test set only once per released model. Jev has only been run on the development sets of these two suites. Brier scores the whole probability distribution, not just the top answer; lower is better.

On new sources Kev-27B is within a point of Jev (0.851 vs 0.857), and Kev-4B and Kev-9B are within four points. We don't know what Jev was trained on, so this isn't a controlled comparison of the two architectures. [What to Expect](#what-to-expect) says where Kev is as good as Jev and where it isn't.

Kev-0.8B, 4B and 9B start from Qwen base models and share one training recipe: a small adapter on a frozen base. Kev-27B starts from Qwen's post-trained release, and we don't know what that was trained on; every one of its weights is fine-tuned, so it ships as 51 GB of full weights rather than an adapter. Each model card has the full recipe, all results, and the earlier versions kept as Hub tags.

## Kev 1.0

The four models above are released together as Kev 1.0. Each Hub repo has a `v1.0` tag, so `--run jaredpalmer/kev-4b@v1.0` always loads the same weights, and the GitHub release [`kev-1.0`](https://github.com/jaredpalmer/kev/releases/tag/kev-1.0) has the 0.8B, 4B and 9B checkpoints with SHA-256 checksums. Kev-27B's 51 GB of weights are too large for a release asset and are on the Hub only.

| Model | Hub revision of the weights | Temperature | Trained on states up to |
|---|---|---|---|
| Kev-0.8B | `9a45d25e` | 2.35 | 7,552 tokens |
| Kev-4B | `139fdd94` | 2.41 | 7,552 tokens |
| Kev-9B | `b5d8c18e` (v2) | 2.19 | 7,552 tokens |
| Kev-27B | `28be62e9` (v2, full weights) | 1.32 | 32,768 tokens |

Kev 1.0 trains nothing new. It fixes the checkpoints, cards, evaluation suites and serving code that the next generation of Kev will be compared against. The [release notes](docs/releases/kev-1.0.md) list what changed since the previous family release and what is known not to work well.

## Quick Start

### Try It in the Browser

The [Hugging Face Space](https://huggingface.co/spaces/jaredpalmer/kev) runs Kev-4B and Kev-0.8B, with nothing to install.

### Run It Locally

> [!WARNING]
> PyPI name collision: `pip install kev` installs a **different package** — [K.E.V. ORM](https://pypi.org/project/kev/) (Brian Jinwright, 2016-2021). This project is **not** published to PyPI. Install from source as below.

You'll need Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). The repo's `.python-version` makes `uv sync` use 3.13; torch has no wheels for 3.14 yet.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

This starts Kev-4B on your machine: CUDA or ROCm if you have a GPU, MLX on Apple Silicon. The first run downloads the adapter and the base model. `--run` also accepts a local checkpoint directory or a Hub revision like `jaredpalmer/kev-4b@qwen3`.

Use `--device cpu` to skip accelerator discovery, or `--device cuda` / `--device mps` to require that accelerator.
The default, `--device auto`, tests an allocation and matrix multiplication in a separate process before loading weights.
It tries CUDA (including ROCm) first; if that probe fails, crashes, or times out, it tries MPS, and only then warns and uses CPU.
An explicitly requested accelerator fails instead of falling back. `kev.train` and `kev.evaluate` take the same `--device` flag
(`--device cuda` there is probed the same way, and the run stops if the probe fails). `KEV_DEVICE_PROBE_TIMEOUT` is the limit
for each probe, in seconds (default 30), so a dead CUDA init followed by an MPS attempt can take twice that.
This checks basic runtime support, not every model kernel.

On Apple Silicon, serving a hybrid checkpoint with the default backend uses MLX and does not run the torch MPS probe.
`KEV_BACKEND=torch` or `KEV_DTYPE=fp32` still probes torch.
CPU serving keeps the fp32 default; `KEV_DTYPE=bf16` reduces weight memory if needed. ROCm uses the eager serving path by default;
the fused-kernel and graph defaults are enabled only on NVIDIA builds.

In another terminal, send it a ticket:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

Example response from Kev-4B, running in bf16 on an Apple M5:

```json
{
  "model": "kev-latest",
  "answers": {
    "department":  { "type": "choice", "choice": "returns", "confidence": 0.21,
                     "probabilities": { "returns": 0.47, "shipping": 0.28, "billing": 0.25 } },
    "escalate":    { "type": "noul", "noul": 0.93 },
    "frustration": { "type": "score", "score": 1.44, "confidence": 0.34,
                     "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                     "probabilities": { "0": 0.00, "1": 0.56, "2": 0.44 } }
  },
  "usage": { "input_tokens": 101, "output_tokens": 161 },
  "latency_ms": 495
}
```

The ticket mentions a return, a late delivery and a billing problem, and the department probabilities say so. That's why Kev returns probabilities instead of a single label: your code can route the confident cases and send the rest to a person.

### Use It From Python

If you already call Jev, point your client at Kev and keep the rest of your code. The TypeSafe SDK is included in `uv sync --extra serve`:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8009",
    model="kev-latest",
)
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    },
)
print(response.nouls["billing"].noul)
print(response.choices["tone"].choice)
print(response.scores["urgency"].score)
```

## Fine-Tune on Your Own Data

The released models were trained on public datasets and generated policy examples. If your questions look different, like your own routing categories, your own escalation rules or another language, a short fine-tune usually helps more than any prompt change. It also fits the temperature to your data, so the confidence you set thresholds on is measured on your own labels.

What to expect: on an example support workload (three questions, 1,050 generated records, 15 minutes on an H100), fine-tuning took Kev-4B from 67.7% to 73.6% accuracy, and from automating 34% of decisions at a 5% error budget to 48% ([details](skills/kev-finetune/README.md#why-fine-tune-at-all)). On real data, one epoch on 5,219 labelled consumer-finance complaints took Kev-4B from 0.804 to 0.904 accuracy on complaints it had never seen. Gains like these are in distribution: they tell you how well Kev learns your task, not how it does on everything else. Size your dataset first. With 400 records, the gain on the example workload was inside the noise.

### With a Coding Agent

```bash
npx skills add jaredpalmer/kev@kev-finetune
```

Then ask your agent to "fine-tune Kev on my support tickets". The [`kev-finetune` skill](skills/kev-finetune/) interviews you, finds the questions your code already asks Jev or TypeSafe, converts the labels you have or generates enough with any LLM to measure a gain, fine-tunes from a released checkpoint on Modal, fits the temperature on a held-out slice, scores the result against the untouched model, deploys an endpoint, and tears everything down at the end. You don't need a local GPU or a clone of this repo. A Kev-4B training run costs about $1 on an H100.

### By Hand

The skill's [README](skills/kev-finetune/README.md) is the same recipe for people: six short standard-library scripts and one Modal app. To train from this repo instead, put your examples in a JSONL file, one request per line. It's the same shape as an API request, plus a `label` on every question:

```jsonl
{"state": {"subject": "Charged twice", "body": "I see two charges for order #4411. Please refund one."},
 "questions": {
   "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems", "access": "Login and account access"}, "label": "billing"},
   "angry":    {"type": "noul",   "instructions": "Is the customer angry?", "label": false},
   "priority": {"type": "score",  "instructions": "How urgent is this ticket?", "criteria": ["low", "normal", "high"], "label": 1}}}
```

For `choice` the label is the option name, for `noul` it's `true` or `false`, and for `score` it's the level's position starting at 0. Keep 10–20% of the file aside for evaluation.

Then start from a released checkpoint with `--init_from`:

```bash
uv run python -m kev.train --data train.jsonl --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
    --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --out runs/mine

uv run python -m kev.benchmark --run runs/mine --data heldout.jsonl --out runs/mine-eval
uv run --extra serve python -m kev.serve --run runs/mine --port 8009
```

`--device cuda` is checked with the same child-process probe before any weights load, and training stops if that check fails. `--device cpu` skips it. `--device auto` tries CUDA, then MPS, then CPU.

`--init_from` loads the adapter and pointer head from the released model before training, so you keep what Kev already knows and add your domain on top. Starting from the base model instead throws that away: in one user's test on 836 support-tool decisions, a fine-tune from the base scored 0.33 on Kev's own evaluation set, against 0.84 for the released model; the same data with `--init_from` kept 0.83 there and reached 0.88 on the new domain. Use a smaller learning rate than the from-scratch recipe (`2e-5` is a good start), and pick `--base` to match the checkpoint you start from; the trainer checks that the base, revision, LoRA rank and head size agree before it loads anything.

`--batch 1 --accum 8` in bf16 fits the 0.8B model on a 4 GB GPU. The benchmark reports accuracy, Brier score and calibration per question type, so you can see which of your questions the fine-tune helped. The checkpoint you started from is recorded in `runs/mine/training_config.json`. On a Mac, run one training job at a time; two jobs on the same Apple GPU are much slower.

## Deploy Your Own Endpoint

To get an HTTPS endpoint instead of a local server, you don't need this repo, just a [Modal](https://modal.com) account:

```bash
pip install modal==1.5.5 && modal setup
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py
KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py
```

That serves Kev-4B on an L40S at `https://<your-workspace>--kev-api.modal.run`, with the same API as above behind `Authorization: Bearer <key>`. It scales to zero when idle, so an unused endpoint costs nothing. The first request after idle waits about 35 seconds for a container to start. `KEV_MODEL=jaredpalmer/kev-9b` serves another model on the GPU that suits it; Kev-27B goes to a B200, falling back to an H200 or H100. If you use a coding agent, `npx skills add jaredpalmer/kev@kev-deploy` does the same and wires the URL into your code. [skills/kev-deploy](skills/kev-deploy/) has the GPU and cost table.

For lower warm HTTP latency, add `KEV_FLASH=1 KEV_REGION=us-east` to the deployment command. This uses Modal's direct server with a 64-request concurrency target instead of the regular web endpoint's 32-request target; it returns a different URL and keeps at least one paid GPU container warm, even when idle. The API and bearer authentication stay the same. [Modal HTTP Performance](#modal-http-performance) separates end-to-end latency from model time.

A model you fine-tuned with the `kev-finetune` skill deploys the same way from its own Modal app (`KEV_SERVE_SECRET=kev-serve-key KEV_SERVE_RUN=<run> modal deploy scripts/kev_modal.py`; see [its deploy guide](skills/kev-finetune/references/deploy.md)). To host Kev on your own machines instead, run `kev.serve` from [Run It Locally](#run-it-locally) on a GPU box with `--host 0.0.0.0` and put it behind your own proxy; [Serving Performance](#serving-performance) says which GPU to pick.

## What to Expect

**Accuracy.** Kev-27B is within three points of Jev, or ahead of it, on 9 of the 11 new-source categories in the chart below. Kev-4B and Kev-9B are about as close on classification-shaped sources like routing, entailment and science questions. Knowledge questions depend mostly on the base model: on MMLU Kev-9B scores 0.73 and Kev-27B matches Jev at 0.90, but on the harder MMLU-Pro Kev-27B scores 0.675 against Jev's 0.840. The smaller models also trail on day-precision date arithmetic.

![Accuracy by source for Kev and Jev](docs/kev-family.png)

**Confidence.** Each checkpoint ships with a fitted temperature, so its probabilities are calibrated by default. As served, Kev-9B puts at least 0.9 probability on a wrong answer for 2.4% of new-source questions, against Jev's 3.7%. Jev still ranks its answers better: at a 5% error budget, Kev-4B, 9B and 27B can automate 0.52–0.69 of new-source decisions, Kev-0.8B 0.14 and Jev 0.70. Check a threshold on your own data before you rely on it.

**Speed.** Kev-4B answers six questions about a new short text in 18.1 ms of model time on an H100 and 41.5 ms on an L40S, and its in-process server handles around 101 requests per second on an H100, before HTTP overhead. On an Apple M5, Kev-4B takes 721 ms for five questions, or 136 ms when the text repeats and comes from the cache. [Serving Performance](#serving-performance) has the GPU measurements and [Modal HTTP Performance](#modal-http-performance) has deployed request latency.

**Length.** Kev-0.8B, 4B and 9B trained mostly on states of up to 384 tokens, with longer ones in their document and skill fine-tunes (up to 7,552 tokens), Kev-27B on states of up to 32,768. The server accepts states of up to 65,536 tokens, and 8,192 more for each question, and refuses a longer one with a 422 instead of cutting it. How far past its training length each model stays accurate is the "Validated context" column in [Models](#models). For Kev-0.8B, 4B and 9B that is 8,192 tokens: at 16k the measurement on real contracts can no longer rule out a drop of more than 3 points, and at 32k all three are measurably less accurate than at 8k. Kev-27B holds to the 65,536-token limit. On real contracts of up to 64k tokens (CUAD) Kev-27B scores 0.874, and its confidence there is less reliable than on short text; its [model card](docs/model-cards/kev-27b.md) has the numbers by length.

## Playground

With the server running, open another terminal. You'll need Node 20.9+:

```bash
cd playground
npm install
npm run dev -- -p 3001
```

Open [localhost:3001](http://localhost:3001), load a preset, and edit the text and questions. Press `⌘↵` to run it. "Packed vs separate" compares asking all questions at once with asking them one at a time. "Permute" runs a Choice question with six option orders. There are also presets for testing question isolation and fake delimiter tokens.

![Kev playground](docs/playground.png)

There's a [chess demo](http://localhost:3001/chess), too. The board is the input, legal moves are Choice options, and a Score question rates the position. You can play against Kev or let it play itself. Games are saved in `localStorage`.

## API

### `POST /v1/systemone`

`state` is the text to evaluate. Each question has instructions and, where needed, a set of answers to choose from.

```jsonc
{
  "state": "…",                          // string | object | array — the content to evaluate
  "model": "kev-latest",
  "questions": {
    "<id>": {                            // you choose the id; the model never sees it
      "type": "noul" | "choice" | "score",
      "instructions": "…",               // string | object | array, optional
      "criteria": …                      // noul: {true?, false?}  choice: {option: description|null}  score: [level, …]
    }
  }
}
```

| Type | Criteria | Answer |
|---|---|---|
| `noul` | Optional descriptions for `true` and `false` | `noul`: probability of yes |
| `choice` | 1–255 option names, each with a description or `null` | `choice`: most likely option; `probabilities` and `confidence` |
| `score` | 1–255 descriptions, ordered from lowest to highest | `score`: mean level index, starting at 0; `legend`, `probabilities`, and `confidence` |

For Choice with `K > 1` options, confidence is `(p_max − 1/K) / (1 − 1/K)`. A single option has confidence 1. Score confidence is `max(0, 1 − E|level − mode| / D)`: `mode` is the most likely level and `D` is the mean distance of a uniform distribution over the levels from its middle (2/3 for three levels), so all probability on one level gives 1 and a uniform or wider spread gives 0. Both formulas are the ones in TypeSafe's reference adapter ([`system-one-adapter`](https://github.com/typesafe-ai/system-one-adapter-python) 0.2.1). Neither field is a measured accuracy rate.

Objects and arrays are converted to labeled text. Delimiter-like strings in user input are escaped before tokenization. Invalid requests return `422`, and so does a state longer than 65,536 tokens: the server never drops part of a document silently, and the error gives the state's token count and the limit. `usage.output_tokens` counts tokens in the serialized answers, not generated tokens.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | Model cards (`name`, `description`, `release_date`) plus the loaded checkpoint's details |
| `POST` | `/v1/systemone/permute` | Run one Choice question with different option orders (`n_perm` 1 to 64, default 6) |
| `POST` | `/v1/systemone/separate` | Run each question in its own forward pass |

A request may carry any number of questions. The server runs them a token budget at a time (one maximal row of 16,384 tokens per forward pass, counting the cached document once per question in that pass), so memory does not grow with the question count and the answers do not depend on the split. Every response carries an `x-typesafe-request-id` header. The server binds to `127.0.0.1` (`--host 0.0.0.0` to accept other machines) and is open by default; set `KEV_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`, as the TypeSafe clients always send it.

| Variable | Effect |
|---|---|
| `KEV_TEMPERATURE=1.0` | Return raw probabilities instead of the calibrated ones |
| `KEV_DATE_FACTS=1` | Append the number of days between any two dates in the state (see [Benchmarks](#benchmarks)) |
| `KEV_TRUNCATE_STATES=1` | Read the first 65,536 tokens of a longer state instead of refusing it; every response then has `truncated` and `usage.state_tokens` / `state_tokens_used` |
| `KEV_DTYPE=fp32` | Serve the exact fp32 path the evaluations use (bf16 is the default on GPUs) |
| `KEV_API_KEY` | Require a bearer key |

## How It Works

Each checkpoint is a rank-16 LoRA adapter and a small pointer head on a Qwen base model. On an attention-only base (Qwen3), the state and questions go into one token sequence:

```text
<state> …state…
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>
```

The attention mask lets a token read the state and its own question, but not other questions or future tokens. Each question's position IDs restart just after the state. This lets the model process the state once and answer each question independently.

Qwen3.5 and Qwen3.8 mix attention layers with Gated DeltaNet layers, which are recurrent and ignore attention masks. For those models, which is every current Kev, each question runs as its own row: the state followed by that question, with the same positions as above. The rows are independent, so isolation is exact, and the server and `DecisionModel.probs()` compute the state once and reuse its cache for every row. `forward()`, which `kev.benchmark` scores and every published number comes from, keeps the plain rows and runs the state once per question; the two agree to fp32 rounding. On attention-only models the rows and the mask above give identical probabilities (`tests/test_model.py`).

Kev-27B uses the same design on `Qwen/Qwen3.8-27B`, with two differences. Its base is Qwen's post-trained release rather than a `-Base` checkpoint, and we don't know what it was post-trained on. And every backbone weight is trained, not just an adapter, and kept in bf16, so the checkpoint is the whole model: 51 GB of bf16 weights plus the pointer head. It serves in bf16 only (about 66 GB resident with the serving buffers), which is why it needs an 80 GB card. On Apple Silicon the MLX backend loads those weights as they are, with no merge (see [Serving Performance](#serving-performance)); we expect that to fit a 96–128 GB Mac but haven't measured it. Its served probabilities stay within 0.022 of the evaluation path on an H200 (`runs/serving-27b-r23`).

The pointer head scores each option's `</opt>` hidden state against the question's `<decide>` hidden state. A softmax turns those scores into probabilities. Because `<decide>` comes last, it can attend to the full option list.

Training uses cross-entropy on the correct answer. The adapter and head are trained together; the rest of the base weights stay fixed (Kev-27B trains them all). Training examples and API requests use the same text format. No Jev outputs were used for training.

Asking questions together or separately produces probabilities within 4e-6 in the fp32 tests. This does **not** mean option order is irrelevant: options within a question can still affect one another. See [the model code](kev/model.py) and [parity tests](tests/test_model.py).

## Training

The released models share one base training set, `decision-v7`: 10,000 examples from ten public datasets, 896 generated policy examples, and 1,680 examples from 60 generated rule structures. Kev-0.8B, 4B and 9B train on it for two epochs with LoRA rank 16 and cross-entropy. The learning rate is `1e-4` for 0.8B and `5e-5` for 4B and 9B. On these hybrid bases the adapter covers the attention, MLP and DeltaNet projections; `kev.train` picks the right targets from the model config.

Kev-0.8B, 4B and 9B then get short follow-up fine-tunes from their released checkpoints, through the same `--init_from` path you'd use for your own data: generated cases that state day counts or have the deciding evidence removed (all three), then real documents and generated skill data (all three; Kev-9B since v2, 2026-09-30). Kev-27B is trained differently. Every weight of the base is fine-tuned for one epoch on eight H200s (`--full_ft 1`, learning rate `2e-6`) on a 145,840-record corpus: Kev's own data, the document, skill and developer-tooling suites, public datasets, licensed task families and generated long-document, tool-routing, agent-log and guardrail records, with states of up to 32,768 tokens. The result is then averaged with the earlier adapter-trained Kev-27B, 0.85 to 0.15. The model cards list every stage with its data and cost.

```bash
# sanity run, ~1 minute
uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke

# the first stage of Kev-0.8B (~20 min on one H100; the Mac path works but is slow for Qwen3.5 bases)
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --epochs 2 --lr 1e-4 --batch 8 --dtype bf16 --p_none_pair 0.25 --device cuda --out runs/kev-0.8b

# the first stage of Kev-4B (one H100 via Modal, ~1 h; see below). Swap in Qwen/Qwen3-4B-Base for the previous generation.
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-4B-Base --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
    --epochs 2 --lr 5e-5 --batch 4 --accum 2 --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --device cuda --out runs/kev-4b
```

Use `uv run python -m kev.train --help` for all training options. The released models don't use the optional `--perm_kl` or `--ord_w` losses. [PLAN.md](PLAN.md) records what was tried, what helped, and what didn't.

### Modal

Each trial gets its own H100. The study keeps running if you disconnect, and you can download the results when it finishes:

```bash
uv run modal token new                                    # once; opens the browser
KEV_GPU=T4 uv run modal run modal_app.py::smoke           # end-to-end check, ~1 minute of GPU

uv run modal deploy modal_app.py                          # once; studies run on the deployed app and survive disconnects
uv run modal run modal_app.py::study \
    --suite evals/v7/decision-v7 --plan experiments/v7-final.json \
    --name my-study --transfer evals/v4/transfer-v4 --budget 30 --timeout 7200
uv run modal run modal_app.py::pull --name my-study       # results -> runs/my-study, ranked
```

[Study plans](experiments/v7-final.json) list training settings. Each trial saves the settings, code hashes, dataset hashes, and results. Choose models using the development results, not the locked test. After choosing a final candidate, you can read its test results once:

```bash
uv run modal run modal_app.py::locked_test --trial my-study/00-trial-0 --name my-candidate   # one read, ever
```

## Benchmarks

The evaluation data under `evals/` is frozen: dataset versions and file checksums are recorded in each manifest. Large files are downloaded from [the Hub mirror](https://huggingface.co/datasets/jaredpalmer/kev-suites) and checked against those hashes. Every model in the tables above is scored on the same items. The numbers in this README and the model cards are checked in CI against the committed reports they come from (`docs/claims.json`, `uv run python scripts/verify_claims.py`).

| Suite | What it measures |
|---|---|
| `decision-v7` | Held-out examples from the ten training datasets, the generated policies and the rule structures ("trained sources") |
| `transfer-v4` | 764 records from datasets and policy and rule types Kev never trained on: QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval, held-out policies and rules ("new sources") |
| `transfer-v9` | `transfer-v4` plus 10-way MMLU-Pro, records buried in unrelated text, and "unknowable" records whose deciding evidence was removed |

```bash
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v4/transfer-v4 --out runs/my-eval      # new sources
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v9/transfer-v9 --out runs/my-eval-v9   # + MMLU-Pro, buried states, unknowable items
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v7/decision-v7 --out runs/my-eval-id   # trained sources
uv run python -m kev.benchmark --remote http://127.0.0.1:8009 --suite evals/v4/transfer-v4 --out runs/my-remote   # any System One endpoint, Jev included
```

These commands use development data. Test data requires `--allow-test`. The benchmark reports accuracy, Brier score, calibration error, the share of decisions you could automate at a 5% error budget, option-order changes, and question isolation. On the unknowable records it reports how often the model still answers with at least 0.9 confidence (Kev-9B 0%, Jev 9%). Published accuracy numbers use fp32 evaluation, not the bf16 serving path. `kev.jev` runs the same questions against Jev through Vercel AI Gateway, and `kev.compare` compares two saved runs with paired bootstrap confidence intervals.

**Calibration.** Each checkpoint stores a temperature, and the pointer head applies it when the model is loaded. Kev-4B (2.41) and Kev-0.8B (2.35) fitted theirs on their in-distribution development sets; Kev-27B (1.32) and Kev-9B (2.19) fitted theirs on held-out datasets they never trained on. A refit of the two smaller models on those held-out datasets was tested and kept neither: it did not improve Kev-4B and made Kev-0.8B worse calibrated on its document and skill suites (the model cards have the numbers). A temperature never changes which answer wins. On new sources it takes Kev-9B's calibration error from 0.103 to 0.041 and its confident errors (wrong answers with probability ≥ 0.9) from 8.2% to 2.4%, below Jev's 3.7%. The accuracy numbers above are the same either way; the Brier numbers are for the raw probabilities. `scripts/calibrate_checkpoint.py` also reports an out-of-fold estimate, so the in-sample fit can be checked against records it didn't see.

**Dates.** Kev can't subtract dates reliably, but it can use a day count it's given. `KEV_DATE_FACTS=1` appends one sentence per pair of dates in the state ("June 26, 2026 is 8 days before July 4, 2026"). On the deadline policy questions this takes Kev-9B from 0.80 to 0.90 (Jev 0.93). None of the tables use it.

**Other people's test sets.** `evals/external/` holds test sets from other projects, converted to this format, with their published live Jev results. Some were scored on earlier versions of the Kev weights, which the Kev column names. Three were removed because they cannot serve as a gate, and the model cards keep the numbers their releases were decided on: scienthoon's synthetic support tickets on 2026-09-27 (templated text; one of its three questions depends on a rule the text does not state), and on 2026-09-30 WANLI (`wanli-v1`, `wanli-v2`: a quarter of the pairs are ones WANLI's two annotators labelled differently, with the gold set to one of them) and TypeSafe's public evals (`typesafe-v1`: the gold is the averaged answer of two closed frontier models, and on 89 questions it cannot tell checkpoints apart).

| Suite | What it is | Jev | Kev |
|---|---|---|---|
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | 144 authored decisions | 0.965 | 0.917 (Kev-9B at `v7-base`) |

SemIf's labels hold up, but it is close to saturated: every Kev-27B checkpoint answers 130 of the 144 correctly, so it is a sanity check, not a way to rank models.

## Serving Performance

Pick the GPU by the model:

| Model | GPU ($/h) | 6 questions, short text | 5 questions, 2,200-token text | Requests/s, 64 clients |
|---|---|---|---|---|
| Kev-0.8B | L4 (0.80) | 22.7 / 16.1 ms | 108.6 / 32.3 ms | 62.8 |
| Kev-4B | L40S (1.95) | 41.5 / 27.7 ms | 145.2 / 43.0 ms | 51.4 |
| Kev-4B | H100 (3.95) | 18.1 / 12.9 ms | 89.4 / 22.5 ms | 100.8 |
| Kev-9B | L40S (1.95) | 66.4 / 42.7 ms | 235.6 / 57.5 ms | 32.7 |
| Kev-9B | H100 (3.95) | 24.0 / 16.6 ms | 88.5 / 26.4 ms | 79.5 |
| Kev-27B | B200 (6.25) | 46.5 / 32.2 ms | 178.0 / 52.1 ms | 44.2 |
| Kev-27B | H200 (4.54) | 67.2 / 50.0 ms | 274.8 / 73.8 ms | 28.6 |
| Kev-27B | H100 (3.95) | 75.0 / 52.0 ms | 277.5 / 79.3 ms | 28.9 |

Times are model time per request (the `latency_ms` the API returns), median of 20, for a new text / the same text again. The server caches the text, so asking more questions about a document you've already sent only pays for the questions. Requests per second are in-process measurements for 64 concurrent clients sending six questions about a new short text each; the server batches them. Kev-27B's B200 and H100 rows were measured on its previous version, the same architecture served in bf16 (`runs/fused-27b-*`); the H200 row is the current checkpoint (`runs/serving-27b-r23`). These measurements exclude HTTP transport; see [Modal HTTP Performance](#modal-http-performance) for deployed latency and throughput.

An L4 is enough for Kev-0.8B but too slow for Kev-4B. The A100 is slower than the L40S here and costs more. Kev-9B needs about 17 GB of GPU memory and Kev-27B 51 GB of weights (about 66 GB with the batching buffers); under load Kev-27B is compute-bound, and a B200, H200 or H100 costs about the same per request. On CUDA, install `flash-linear-attention` for the Qwen3.5 models (`kev_serve.py` and the Modal images already do).

On Apple Silicon, `uv sync --extra serve` installs [MLX](https://github.com/ml-explore/mlx-lm) and the server uses it automatically. Five questions about a ~270-token text on an M5 (32 GB):

| Model | New text | Same text again |
|---|---|---|
| Kev-0.8B | 149 ms | 28 ms |
| Kev-4B | 721 ms | 136 ms |

Long documents are read into the cache 1,024 tokens at a time, so memory stays close to the weights. With a 65,000-token document, Kev-0.8B takes 21.2 s the first time and 202 ms after that, at a 3.8 GB peak, and Kev-4B 84.5 s and 716 ms at 13.0 GB (`runs/mlx-long-states`; the model cards have every length). Kev-9B has not been measured this way yet.

Adapter checkpoints are folded into the base as they load, which briefly holds a second copy of the weights. Full-weight checkpoints like Kev-27B load as saved, with nothing merged, so loading needs only the weights. We checked this on Kev-4B written out as full bf16 weights: loading peaked at 8.4 GB for 8.4 GB of weights, against 15.9 GB for the adapter path. Its answers matched the adapter path exactly once both hold the same bf16 values, and stayed within 0.015 of the fp32 path on 60 questions (`runs/mlx-full-4b`). Kev-27B's weights are 51 GB. By the same measurements it needs about 51 GB plus working memory, so a 64 GB Mac is borderline and a 96–128 GB Mac should fit. We haven't run it on a Mac that large yet. Kev-27B's first version, an adapter, did run this way on a 128 GB M5 Max, matching the published accuracy (thanks to Sean Connelly, [#175](https://github.com/jaredpalmer/kev/pull/175)).

The server runs in bf16 on GPUs and Macs. Its probabilities differ from the fp32 path the published evaluations use by at most about 0.03 on a GPU and 0.05 on a Mac, and the top answer changes on about one question in 300. Set `KEV_DTYPE=fp32` for the exact path. `/v1/models` reports the backend and precision in use. `uv run modal run modal_app.py::serving --run jaredpalmer/kev-4b --gpu L40S --name <name>` measures a row of the table on your own account (the rows above: `runs/serve-*`, `runs/grouping-4b-h100`, `runs/fused-27b-*`, `runs/serving-27b-r23`).

### Modal HTTP Performance

End-to-end warm HTTP latency, including transport and queueing, is different from model time. This paired run used Kev-4B, one H100 and four CPUs per endpoint in `us-east`, kept-alive clients in the same region, bearer authentication, and six questions about a new short text per request. After warm-up, each mode handled 128 timed requests at one client, 256 at 32 clients, and 512 at 64 clients:

| Modal endpoint | Concurrent clients | HTTP p50 (ms) | HTTP p99 (ms) | Requests/s |
|---|---|---|---|---|
| Regular web | 1 | 102.1 | 112.2 | 9.8 |
| Direct server | 1 | 36.9 | 40.4 | 26.7 |
| Regular web | 32 | 902.0 | 1177.7 | 35.6 |
| Direct server | 32 | 598.7 | 911.9 | 51.2 |
| Regular web | 64 | 956.4 | 3402.7 | 53.4 |
| Direct server | 64 | 620.9 | 2823.4 | 73.2 |

The direct server bypasses Modal's Function input queue; it does not change the model path or the in-process measurements above. All timed HTTP requests succeeded, sample answer payloads matched, and bearer-authentication, validation and TypeSafe SDK checks passed. These are short warm runs, not a tail-latency SLA: new batch shapes still captured CUDA graphs during the measurement, and latency depends on the workload and client location. The concurrency target controls scaling, not requests per second.

Enable the direct path with `KEV_FLASH=1 KEV_REGION=us-east` (also `KEV_GPU=H100` to use the GPU measured here); it keeps at least one paid container warm instead of scaling to zero. [The report](runs/modal-http-readme-4b-h100/report.json) includes checkpoint and kernel provenance, process/model latency and graph counts. Reproduce it with `uv run --extra serve modal run scripts/modal_http_bench.py --out runs/modal-http-readme-4b-h100 --levels 1,32,64`; [the deployment guide](skills/kev-deploy/SKILL.md#throughput) covers the earlier comparison and deployment-script smoke check.

## Limitations

- Calibration is a single temperature. It can't reorder confidences, so the share of new-source decisions you can automate at a 5% error budget (0.52–0.69 for Kev-4B, 9B and 27B) is still below Jev's 0.70. Test a probability threshold on your own data before you rely on it.
- Knowledge questions are set by the base model. MMLU is 0.73 for Kev-9B against Jev's 0.90, and MMLU-Pro 0.59 against 0.84.
- Fine-tuning can make the base model worse at individual tasks. Date arithmetic was the clearest case ([issue #8](https://github.com/jaredpalmer/kev/issues/8)); training on stated day counts plus `KEV_DATE_FACTS=1` recovers it.
- Changing option order can change an answer. Question isolation doesn't prevent this.
- Kev-0.8B, 4B and 9B trained mostly on at most 384 state tokens and 1,024 tokens for the state plus one question (their document and skill fine-tunes on states of up to 7,552 tokens), Kev-27B on states of up to 32,768 tokens. Serving allows a 65,536-token state; the validated context length of each model is in [Models](#models).
- On a Mac, answers take hundreds of milliseconds, not tens. Kev-27B needs an 80 GB GPU. On a Mac it needs about 51 GB plus working memory; we expect a 96–128 GB Mac to fit it but haven't measured one.
- Kev-27B starts from a post-trained model whose training data we don't know.

## Development

```bash
uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py \
    tests/test_documents_tools.py tests/test_hard_v1.py tests/test_devtools_v1.py tests/test_breadth_v1.py tests/test_rounds.py tests/test_skill_scripts.py -q   # no weights, no server; what CI runs
KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q   # against a running server
cd playground && npm run lint && npx next typegen && npx tsc --noEmit -p .
```

The API tests run TypeSafe's example requests and the official SDK against your local server. [PLAN.md](PLAN.md) is the research plan: what we have learned, the rules every experiment follows, and one line per round. The full log (every experiment, the criteria set before it ran, and how it came out) is at the git tag `research-archive-2026-09-24`.

<details>
<summary>Previous generation (Qwen3) and the prototype</summary>

The first Kev family used Qwen3 bases with the same data and settings. Those weights stay published and run on plain PyTorch on a Mac, but they are no longer developed.

| Model | Base | Accuracy: Trained Sources | Accuracy: New Sources | Brier: New Sources | Model Card |
|---|---|---|---|---|---|
| Kev-0.6B (Qwen3) — `jaredpalmer/kev-0.6b` | Qwen3-0.6B-Base | 0.801 / 0.808 | 0.620 / 0.642 | 0.536 / 0.483 | [Details](docs/model-cards/kev-0.6b-qwen3.md) |
| Kev-4B (Qwen3) — `jaredpalmer/kev-4b@qwen3` | Qwen3-4B-Base | 0.854 / 0.856 | 0.790 / 0.806 | 0.328 / 0.294 | [Details](docs/model-cards/kev-4b-qwen3.md) |
| Kev-8B (Qwen3) — `jaredpalmer/kev-8b` | Qwen3-8B-Base | 0.863 / 0.870 | 0.796 / 0.780 | 0.337 / 0.327 | [Details](docs/model-cards/kev-8b-qwen3.md) |

The original [Kev-0.5B](https://huggingface.co/jaredpalmer/kev-0.5b) used Qwen2.5-0.5B and is kept for reference; see its [model card](docs/model-cards/kev-0.5b.md).

</details>

<details>
<summary>Troubleshooting</summary>

- If MPS runs out of memory during training, check that you're running only one job. Don't enable `output_hidden_states` or add tokens with peft's `trainable_token_indices`; both have caused memory problems here.
- If the playground loads but buttons don't work, use `localhost:3001`. Next.js checks development hostnames. Other hosts need an entry in `allowedDevOrigins` in `playground/next.config.ts`.
- If dataset loading reports `Dataset scripts are no longer supported`, use `legacy-datasets/banking77`. This repo already uses it.

</details>

## Authors

- Jared Palmer ([@jaredpalmer](https://github.com/jaredpalmer))

Built with [Devin](https://devin.ai). Thanks to [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) for the architecture write-up, [TypeSafe](https://docs.typesafe.ai/api) for the API design, and [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B-Base) for the base models.

Related work: [Hydragen](https://arxiv.org/abs/2402.05099), [DeFT](https://arxiv.org/abs/2404.00242), [FIRST](https://arxiv.org/abs/2406.15657).

## License

[Apache-2.0](LICENSE). The Qwen3, Qwen3.5 and Qwen3.8 base models are also Apache-2.0. Training datasets have their own licenses; see the [model cards](docs/model-cards/).
