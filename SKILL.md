---
name: finetune-inkling-2-small
description: Fine-tune Thinking Machines' Inkling 2 Small (594B-A17B MoE) with LoRA on Tinker, then use it on Tinker, self-host it or publish it. Use when the user wants to fine-tune, evaluate, merge, publish or serve thinkingmachines/Inkling-2-Small. Covers the H200K tokenizer and tml_v0 rendering at a fixed reasoning effort, dataset validation and sizing, the adapter-shape decision (attention+unembed LoRA that merges exactly into the released NVFP4/MXFP8 checkpoint vs all-linear LoRA that stays on Tinker), learning rate and cost, behaviour probes and a self-distilled conversational slice, moving a 64K adapter to the 256K variant, the exact merge, token-level merge verification, SGLang/vLLM serving and honest evaluation.
license: MIT
compatibility: Python 3.11+, tinker-cookbook>=0.6.0 and tml-renderers>=0.4.0, a Tinker API key, optionally Weights & Biases and Hugging Face tokens. Dataset steps run offline; training, sampling, probes and transplants call Tinker and cost money. The merge needs ~790 GB of disk (or Modal); serving needs >=500 GB of GPU memory.
metadata:
  author: Migel Tissera
  version: "1.0.0"
  source: https://github.com/drost-ai/finetune-inkling-2-small
---

# Fine-tune Inkling 2 Small on Tinker

You are helping someone fine-tune `thinkingmachines/Inkling-2-Small` with a LoRA on
[Tinker](https://thinkingmachines.ai/tinker), then use it the way they choose: on
Tinker, on their own GPUs, or published. Many users keep their model private; never
upload anything unless they ask. This skill is the recipe that produced Tess-4.5-594B-A17B (88.2% on
SWE-Bench Pro v2 Hard at xhigh). It starts from a dataset; for turning coding-agent
transcripts into one, use the companion skill
[tinker-ai-skill](https://github.com/drost-ai/tinker-ai-skill), whose rows this skill
accepts as they are.

Run commands from this skill's directory. Every script has an offline default and
explains its flags in `--help`. Read the matching file in `references/` when a step
needs judgement.

## Before you start

1. **Environment.** `uv venv && uv pip install -r requirements.txt`. The pins matter:
   older `tml-renderers` silently renders Inkling 2 with the wrong tokenizer.
2. **Keys.** Copy `.env.example` to `.env`: `TINKER_API_KEY`, `WANDB_API_KEY`
   (optional), `HF_TOKEN` (only to publish). Never print or commit them.
3. **Money.** Training costs train tokens x $1.50 per million on the 64K variant
   (Nov 2026 list price). ~1,000 rows of up to 64K tokens is ~47M tokens, ~$70 per
   epoch. Run every script's offline mode first and **get the user's yes on the
   printed estimate before anything that calls Tinker.**
4. **Decide two things with the user now**, because they change the run:
   - Will they ever need the weights outside Tinker (self-hosting, or publishing)?
     Then the adapter must be `release` (attention + unembed). Tinker-only use can
     take `full`.
     `references/adapter-shape.md`.
   - Which system prompt, tool set and reasoning effort will the model be served
     with? Training rows must carry exactly those.

## The workflow

### Step 1. Get rows

One JSON object per line: `{"messages": [...], "tools": [...], "group": "<session id>"}`.
Messages are OpenAI-style; assistant thinking goes in a leading
`{"type": "thinking", "thinking": "..."}` content part or a `reasoning_content`
field; `tools` are OpenAI function definitions; `group` keeps windows of one session
on one side of the split. To try everything without data or spending:

```
python scripts/make_synthetic_rows.py --out demo/rows.jsonl --sessions 40 --long-results
```

### Step 2. Validate, measure and split

```
python scripts/prepare_dataset.py --input data/rows.jsonl --name my-run
```

Renders every row with the exact training rendering (`tml_v0`, H200K tokenizer,
tools declared, effort 0.9, loss on every assistant message including thinking),
fails on rows over the budget (65,536 tokens for the 64K variant) or structurally
broken, drops harness noise it can identify, splits by group with a fixed seed, and
writes `data/datasets/my-run.{train,val}.jsonl` plus a report with token counts,
the cost per epoch, sha256s and the rendering provenance. **Read the ADVICE lines.**
Rows chunked for another model must be re-chunked for this tokenizer. Details:
`references/model-facts.md`, `references/training.md`.

### Step 3. Add a conversational slice (when ADVICE says most rows open with a tool call)

```
python scripts/conversational_slice.py --source-rows data/rows.jsonl --n 150 --out data/conversational --dry-run
python scripts/conversational_slice.py --source-rows data/rows.jsonl --n 150 --out data/conversational
python scripts/prepare_dataset.py --input data/rows.jsonl --input data/conversational.jsonl --name my-run
```

Samples greetings, questions and planning talk from the base model itself, with your
system prompt and tools, so the fine-tune does not answer "Hey there!" with a tool
call. A few dollars. Why and how: `references/behaviour.md`.

### Step 4. Train

```
python scripts/probe.py --dataset data/datasets/my-run --model thinkingmachines/Inkling-2-Small -n 8   # base baseline
python scripts/train.py --dataset data/datasets/my-run                 # offline smoke test: read the decoded row
python scripts/train.py --dataset data/datasets/my-run --train         # after the user approves the cost
```

Defaults are the shipped recipe: `--adapter release` (rank 32, attention +
unembed, experts frozen), LR 1e-4, batch 8, one epoch, linear decay, eval and save
every 20 steps, W&B on. Tinker's `get_lr` has no Inkling calibration; if budget
allows run a `--learning-rate 3e-4` sibling into a separate `--log-dir`. Watch
`test/nll` at steps 20-40: a dip then a rise means the LR is too hot.

### Step 5. Probe every checkpoint

```
python scripts/probe.py --dataset data/datasets/my-run --model tinker://<run-id>:train:0/sampler_weights/000020
```

Sampler paths are in `runs/<run>/checkpoints.jsonl`. Greeting and no-tool question
must stay text, the task must call a tool. Compare with the base baseline; use
`-n 12` before concluding from a single miss. Pick the checkpoint by behaviour and
your eval, not by NLL.

### Step 6. Use it on Tinker, at 256K

```
python scripts/transplant_256k.py --run-dir runs/<run> --checkpoint final --needle 130000 250000
```

Loads the 64K-trained adapter onto `thinkingmachines/Inkling-2-Small:peft:262144`
in seconds, no retraining. The caps count prompt **plus** output, so coding agents
need this. Point any OpenAI- or Anthropic-compatible client at Tinker with the
printed sampler path as the model name, `reasoning_effort` set, compaction below the
cap. `references/long-context.md`.

### Step 7. Export and merge exactly (to self-host or publish)

Skip steps 7-9 if the model stays on Tinker.

```
tinker checkpoint download tinker://<run-id>:train:0/weights/final -o ./adapters
python scripts/merge_lossless.py selftest
python scripts/merge_lossless.py check --adapter-dir ./adapters/<dir> --base-index ./base/model.safetensors.index.json
hf download thinkingmachines/Inkling-2-Small --revision <commit> --local-dir ./base
python scripts/merge_lossless.py merge --base-dir ./base --adapter-dir ./adapters/<dir> --out-dir ./merged
```

`W <- bf16(fp32(W) + (alpha/r)·B@A)` on the BF16 attention and unembedding tensors;
everything else byte-identical, every shard the same size as the base. Refuses an
adapter that touches the 4-bit experts. Needs ~790 GB of disk; without it, run the
same stages on Modal with `scripts/modal_app.py`. Optionally publish the adapter
alone with `tinker checkpoint push-hf`. `references/export-merge-serve.md`.

### Step 8. Verify the merge, token by token

Serve `./merged` (step 9), then score the same validation rows through Tinker and
through the server with `scripts/merge_check.py` (render, tinker, sglang, compare),
and probe the server with `probe.py --base-url`. The merged model must sit at kernel
noise from the Tinker adapter and clearly away from the base. Only then serve it,
or, if the user wants it published, upload it (`hf upload`, private first).

### Step 9. Serve

SGLang or vLLM with the base model's recipes (reasoning parser `inkling`, tool
parser `inkling2` on SGLang / `inkling` on vLLM), at least ~500 GB of GPU memory
(two B300, four B200, four H200), context up to 1,048,576. DSpark speculative
decoding roughly doubles per-stream speed but shrinks the KV pool; size concurrency
to it. No GGUF: llama.cpp does not support the architecture yet.

### Step 10. Evaluate honestly

Run the base through your own harness first, at the same effort. Evaluate at both
`high` and `xhigh`: effort moved our fine-tune from 70.6% to 88.2% on the same 51
tasks while the base stayed at 78.4%. Report counts, the harness, and both base
numbers (card and yours). `references/evaluation.md`.

## Rules that saved us the most time

- Resolve the renderer from the model name and assert the H200K tokenizer. Never
  reuse rows chunked or rendered for another model.
- Decide `release` vs `full` before training. A `full` adapter cannot leave Tinker
  without a lossy requantization of the experts.
- Train with the exact system prompt, full tool set and effort you will serve with.
- Probe the base, then every checkpoint, with the real tool set.
- Train at 64K, transplant to 256K; never point an agent at the 64K cap.
- Prove the merge token by token before serving or publishing it. Upload only when
  the user asks, and private first.
- Sweep reasoning effort at evaluation and benchmark the base on your harness.
- Never commit `.env`, `data/`, `runs/`, adapters or merged weights; the
  `.gitignore` covers them.
