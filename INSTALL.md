# Install & Use

Open Fusion is a small Python package with **zero required dependencies** (stdlib
only). It exposes a CLI (`open-fusion`) and a library API (`from open_fusion import fuse`).

## 1. Prerequisites

- Python 3.10+
- An API key for an OpenAI-compatible gateway. The default targets **OpenRouter**:
  ```bash
  export OPENROUTER_API_KEY="sk-or-..."
  ```
  To use a different gateway (or direct OpenAI), set:
  ```bash
  export OPEN_FUSION_BASE_URL="https://your-gateway/v1"
  export OPEN_FUSION_API_KEY="..."
  ```

> **Model slugs**: presets in `config/presets.yaml` use OpenRouter-style slugs
> (e.g. `anthropic/claude-opus-4.1`). Slugs change over time — verify current ones
> at https://openrouter.ai/models and edit the preset or pass `--panel`/`--judge`.

## 2. Install

```bash
cd open-fusion-skill
pip install -e .          # editable; or: pip install .
```

This registers the `open-fusion` console script. Verify:

```bash
open-fusion --help
python tests/test_open_fusion.py        # 31 offline tests, no network/key needed
```

## 3. Use the CLI

```bash
# default Quality panel (Opus + GPT, judged by Opus)
open-fusion "What are the strongest arguments for and against a carbon tax?"

# Budget panel (~half the cost of a single frontier model)
open-fusion "Compare ridge, lasso, and elastic-net regression." --preset budget

# custom panel + judge
open-fusion "Design a rate limiter for a multi-tenant API." \
    --panel "anthropic/claude-opus-4.1,openai/gpt-5,deepseek/deepseek-chat" \
    --judge "anthropic/claude-opus-4.1" --show-analysis

# enable grounded panels (needs a search key; see section 5)
export EXA_API_KEY="..."
open-fusion "Latest results on speculative decoding throughput." --tools \
    --exclude-domains "en.wikipedia.org"

# machine-readable output
open-fusion "..." --json
```

The final answer prints to **stdout**; telemetry (and `--show-analysis`) prints to
**stderr**, so you can pipe the answer cleanly:
`open-fusion "..." > answer.md`.

## 4. Use in an agent (Claude Code / Codex)

Both Claude Code and Codex are command-line agents that read a skill/instructions
file and then call tools. Open Fusion plugs in as a CLI the agent invokes when a
prompt warrants multi-model deliberation.

### Claude Code

1. Install the package (section 2) so `open-fusion` is on PATH.
2. Make the skill discoverable. Either:
   - **Project skill**: copy this folder into `.claude/skills/open-fusion/` in your
     repo (the `SKILL.md` here is the manifest), or
   - **Personal skill**: copy into `~/.claude/skills/open-fusion/`.
3. That's it. When you ask Claude Code something like *"get a multi-model consensus
   on this architecture decision"* or *"fuse a few models on this question"*, it
   reads `SKILL.md` and runs `open-fusion "..."`. The `description` field in
   `SKILL.md` is tuned to trigger on fusion/panel/consensus/second-opinion phrasing.

Quick check inside Claude Code:
```
> Use open-fusion to get a budget-panel answer on: when does CRDT beat OT for collab editing?
```

### Codex (OpenAI CLI agent)

Codex reads repo instructions (e.g. `AGENTS.md`) and runs shell commands. Add a
short note pointing at the tool:

```markdown
## Multi-model deliberation
For high-stakes/ambiguous questions, run:
`open-fusion "<question>" --preset quality`   (or --preset budget for cost)
Read SKILL.md in open-fusion-skill/ for when to use it and flags.
```

Then Codex will call `open-fusion` when appropriate. Same binary, same flags.

### As a tool the *model itself* decides to call

The Fusion design is "the model calls it only when the question warrants it." To
mirror that, register `open-fusion` as a function/tool in your own agent and let
the model invoke it with a `question` argument. Keep it selective — fusion is
~2–3× slower and bills additively; it's for research/decisions, not chit-chat.

## 5. Optional: grounded panels (tools)

Tools are **off by default** so the skill works with just a model API key. Enable
with `--tools` (CLI) or `tools_enabled=True` (library). Then:

- `web_fetch` — works with no extra key (stdlib).
- `web_search` — set `EXA_API_KEY` (preferred; supports domain exclusion) or
  `BRAVE_API_KEY`. Without one, search returns a "not configured" result and panels
  proceed without it.
- `bash` — **disabled** unless `OPEN_FUSION_ENABLE_BASH=1`. It runs model-generated
  shell commands locally; only enable in a sandbox you trust.

Always pass `--exclude-domains` when benchmarking to avoid answer/rubric
contamination (see `references/fusion-mechanism.md` §5).

## 6. Library API

```python
import asyncio
from open_fusion import fuse, load_preset, FusionConfig, ModelSpec

async def main():
    cfg = load_preset("budget")            # or build FusionConfig directly
    result = await fuse("Your question", cfg)
    print(result.text)                     # final answer
    print(result.analysis.to_dict())       # structured deliberation
    print(result.telemetry)                # completions, tokens, latency, status

asyncio.run(main())
```

`FusionResult.status` is one of `ok`, `error`, `judge_fallback`. Always check it.
