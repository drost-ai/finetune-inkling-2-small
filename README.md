# finetune-inkling-2-small

**Fine-tune Inkling 2 Small on Tinker and ship it as open weights. The recipe behind Tess-4.5-594B-A17B.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-agentskills.io-black)](https://agentskills.io)
[![Works with](https://img.shields.io/badge/works%20with-Claude%20Code%20%C2%B7%20Codex%20%C2%B7%20OpenCode-brightgreen)](#install-the-skill)
[![Model](https://img.shields.io/badge/model-Inkling%202%20Small-7b3fe4)](https://huggingface.co/thinkingmachines/Inkling-2-Small)

Inkling 2 Small is a 594B-parameter Mixture-of-Experts model with 17B active
parameters, a million-token window and effort-controlled reasoning. You can
fine-tune it on Tinker in an afternoon. Turning that fine-tune into a model people
can download, serve and trust has traps that cost us real runs:
- a tokenizer that silently changed,
- experts stored in 4-bit that a normal LoRA cannot be merged into,
- a 64K cap that counts output tokens,
- a reasoning effort that moved our benchmark score by 18 points.

This skill hands your coding agent the whole path, with each of those decisions
made explicit and every step scripted. Install it and say *"fine-tune Inkling 2
Small on data/rows.jsonl and get it ready to release"*. The agent takes it from
there, and stops before anything that costs money.

## Proven on a release

We trained [Tess-4.5-594B-A17B](https://huggingface.co/migtissera/Tess-4.5-594B-A17B)
([LoRA](https://huggingface.co/migtissera/Tess-4.5-594B-A17B-LoRA)) with exactly this
method:

| | Tess-4.5-594B-A17B | Inkling 2 Small |
|---|:---:|:---:|
| SWE-Bench Pro v2 Hard (xhigh) | **88.2%** | 86.3% card · 78.4% our harness |
| LongBench v2, 1M context | 70.2% | 70.6% |
| AA-LCR v1.1 | **85.0%** | 83.3% |
| Output tokens on those SWE-Bench tasks | **−9%** | baseline |

One epoch, 46.9M tokens, $70 of training, an exact merge into the released
checkpoint, and a merged model verified token by token against the adapter before
it was published.

## What makes Inkling 2 Small different, and what the skill does about it

| The trap | What happens | What the skill does |
|---|---|---|
| **Tokenizer.** Inkling 2 uses H200K; earlier Inklings use o200k | An old renderer package used the wrong one without an error; "64K" rows overflowed by up to 16K tokens | One shared rendering path that asserts H200K; every row measured with it |
| **4-bit experts.** The open checkpoint stores experts in NVFP4/MXFP8 | A LoRA on the experts cannot be merged without lossy requantization | `--adapter release`: attention + unembed LoRA, merged exactly in FP32; a merge script that refuses anything else |
| **Caps count output.** 65,536 / 262,144 on Tinker | An agent's first request plus 16K of output room is rejected | Train at 64K, transplant the adapter to 256K in seconds, no retraining |
| **Agent data says "start working".** 112 of 113 sessions opened with a tool call | The fine-tune calls tools on "Hey there!" | A self-distilled conversational slice from the base, and a probe for every checkpoint |
| **Effort is a dial.** Rows are trained at one effort | 70.6% at `high`, 88.2% at `xhigh`, same model, same tasks | Effort fixed in training, swept in evaluation |
| **"Lossless" is a claim** | A wrong scale or a skipped shard looks fine until it doesn't | Per-token logprob comparison of the served merge against the Tinker adapter |

## The workflow

| Step | What happens | Script |
|---|---|---|
| 1. Rows | `{"messages", "tools", "group"}` JSONL, from [tinker-ai-skill](https://github.com/drost-ai/tinker-ai-skill) or your own pipeline | `make_synthetic_rows.py` for a dry run |
| 2. Prepare | Normalize, render every row with the training rendering, enforce the budget, split by session, report cost and provenance | `prepare_dataset.py` |
| 3. Converse | Sample a conversational slice from the base so the model keeps talking when it should | `conversational_slice.py` |
| 4. Train | LoRA on Tinker with the chosen adapter shape, smoke test and cost first, W&B on | `train.py` |
| 5. Probe | Greeting, no-tool question, task: on the base and every checkpoint | `probe.py` |
| 6. 256K | Move the adapter to the 256K variant, needle-test it | `transplant_256k.py` |
| 7. Merge | Exact merge into the released checkpoint, locally or on Modal | `merge_lossless.py`, `modal_app.py` |
| 8. Verify | Token-level comparison of the served merge against the adapter | `merge_check.py` |
| 9. Serve | SGLang or vLLM with the base recipes, DSpark trade-offs | `references/export-merge-serve.md` |
| 10. Evaluate | Base on your harness first, effort swept | `references/evaluation.md` |

## Quick start: five minutes, no data, no spend

```
git clone https://github.com/drost-ai/finetune-inkling-2-small.git ~/.claude/skills/finetune-inkling-2-small
cd ~/.claude/skills/finetune-inkling-2-small
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

python scripts/make_synthetic_rows.py --out demo/rows.jsonl --sessions 40 --long-results
python scripts/prepare_dataset.py --input demo/rows.jsonl --name demo --out-dir demo/datasets --val-fraction 0.1
python scripts/train.py --dataset demo/datasets/demo          # offline smoke test: decoded row, steps, cost
python scripts/merge_lossless.py selftest                      # the exact-merge guarantees, on a toy checkpoint
```

Then put your keys in `.env`, point step 2 at your own rows, and let your agent
follow `SKILL.md`.

## Install the skill

Clone it into your agent's skills directory. The directory must be named
`finetune-inkling-2-small`, which is what `git clone` gives you by default:

| Agent | Personal (all projects) | Project (this repo) |
|---|---|---|
| Claude Code | `~/.claude/skills/finetune-inkling-2-small/` | `.claude/skills/finetune-inkling-2-small/` |
| OpenAI Codex | `~/.agents/skills/finetune-inkling-2-small/` | `.agents/skills/finetune-inkling-2-small/` |
| OpenCode | `~/.config/opencode/skills/finetune-inkling-2-small/` | `.opencode/skills/finetune-inkling-2-small/` |

```
git clone https://github.com/drost-ai/finetune-inkling-2-small.git ~/.claude/skills/finetune-inkling-2-small
git clone https://github.com/drost-ai/finetune-inkling-2-small.git ~/.agents/skills/finetune-inkling-2-small
git clone https://github.com/drost-ai/finetune-inkling-2-small.git ~/.config/opencode/skills/finetune-inkling-2-small
```

Invoke it with `/finetune-inkling-2-small` in Claude Code; Codex and OpenCode pick it
up from its description. Any agent that supports [Agent Skills](https://agentskills.io)
works the same way.

## What it costs

Tinker list prices, November 2026
([current prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)):

| | Cost |
|---|---|
| Training, 64K variant | $1.50 per million tokens: ~1,000 rows of up to 64K ≈ 47M tokens ≈ **$70 per epoch** |
| Training, 256K variant | $3.00 per million tokens (rarely needed: transplant instead) |
| Conversational slice, 150 conversations | a few dollars |
| Probes, transplant, needle test | cents |
| Exact merge | CPU only: minutes of compute plus ~790 GB of disk for a few hours |
| Serving | your GPUs: ≥500 GB of GPU memory (two B300, four B200, four H200) |

The dataset steps, smoke tests and the merge self-test are free and run on a laptop.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip)
- `tinker-cookbook>=0.6.0`, `tml-renderers>=0.4.0` (see `requirements.txt`)
- A [Tinker](https://tinker.thinkingmachines.ai) API key; [Weights & Biases](https://wandb.ai/authorize) optional
- To publish: a Hugging Face token with write access; to merge: ~790 GB of disk or a
  [Modal](https://modal.com) account; to self-host: ≥500 GB of GPU memory

## Layout

```
SKILL.md                          the workflow your agent follows
scripts/inkling2.py               the one rendering path, constants, row normalization
scripts/make_synthetic_rows.py    step 1: synthetic rows for a dry run
scripts/prepare_dataset.py        step 2: validate, measure, split, report
scripts/conversational_slice.py   step 3: self-distilled conversational rows
scripts/train.py                  step 4: LoRA on Tinker (release / full adapter)
scripts/probe.py                  step 5: behaviour probe, any OpenAI-compatible endpoint
scripts/transplant_256k.py        step 6: 64K adapter onto the 256K variant, needle test
scripts/merge_lossless.py         step 7: exact merge, check, selftest
scripts/modal_app.py              step 7 on Modal: download, merge, upload, serve
scripts/merge_check.py            step 8: token-level verification of the merge
references/                       model facts, adapter shape, training, behaviour,
                                  long context, export/merge/serve, evaluation, lessons
```

## Author and license

Written by [Migel Tissera](https://huggingface.co/migtissera), creator of the Tess
models. MIT license. Inkling 2 Small, Tinker and tinker-cookbook are products of
Thinking Machines Lab; this skill is an independent guide built on them.

If this saved you a run, star the repository and share what you trained.
