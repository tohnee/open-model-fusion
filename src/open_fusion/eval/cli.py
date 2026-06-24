"""
open_fusion.eval.cli - the `open-fusion-eval` console script.

Examples:
    # zero-key demo (canned answers) -> see the report format + verdict
    open-fusion-eval --demo

    # real run: budget panel vs a solo baseline, LLM judge, 3 passes, grounded
    open-fusion-eval --tasks sample --preset budget \
        --baseline anthropic/claude-opus-4.8 \
        --grader llm --grader-model google/gemini-3.1-pro-preview \
        --passes 3 --tools --md report.md

    # offline rule grader against your own DRACO-style file (still needs a key to
    # run the panel/baseline models, but no judge model)
    open-fusion-eval --tasks my_tasks.json --grader rule --baseline openai/gpt-5.5
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from ..client import ModelClient
from ..config import FusionConfig, ModelSpec, from_cli, load_preset
from . import load_tasks
from .grader import LLMJudgeGrader, RuleGrader
from .harness import evaluate
from .report import render_json, render_markdown


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="open-fusion-eval",
        description="DRACO-style evaluation: Fusion vs a solo baseline, with a verdict.")
    p.add_argument("--tasks", default="sample", help="'sample' or path to a DRACO-style JSON file.")
    p.add_argument("--suite", choices=["sample", "draco", "longhorizon", "semiconductor", "fusion", "all"],
                   default=None, help="Built-in task suite (overrides --tasks). Use fusion for a small Fusion-capability smoke test.")
    p.add_argument("--preset", choices=["quality", "budget"], default=None)
    p.add_argument("--panel", default=None, help="Comma-separated panel slugs (overrides preset).")
    p.add_argument("--judge", default=None, help="Judge/synthesizer slug.")
    p.add_argument("--baseline", default=None, help="Solo baseline model slug to compare against.")
    p.add_argument("--grader", choices=["rule", "llm"], default="llm")
    p.add_argument("--grader-model", dest="grader_model", default="google/gemini-3.1-pro-preview")
    p.add_argument("--passes", type=int, default=3, help="Independent judging passes per task.")
    p.add_argument("--tools", action="store_true", help="Enable grounded panels (web/exec).")
    p.add_argument("--exclude-domains", dest="exclude_domains", default=None)
    p.add_argument("--demo", action="store_true", help="Run offline with canned answers (no key).")
    p.add_argument("--md", default=None, help="Write the markdown report to this path.")
    p.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tasks = load_tasks(args.suite or args.tasks)

    if args.demo:
        from . import run_demo
        report = run_demo(tasks)
    else:
        # build fusion config
        if args.panel:
            panel = [ModelSpec(s.strip()) for s in args.panel.split(",") if s.strip()]
            judge = ModelSpec(args.judge or panel[0].slug)
            cfg = FusionConfig(panel=panel, judge=judge)
            # 性能优化 O1：3+ panel 默认 fast_majority_k = N-1，与 load_preset 保持一致。
            if len(cfg.panel) >= 3 and cfg.fast_majority_k is None:
                cfg.fast_majority_k = len(cfg.panel) - 1
        else:
            cfg = load_preset(args.preset or "quality")
            if args.judge:
                cfg.judge = cfg.caller = ModelSpec(args.judge)
        cfg.tools_enabled = bool(args.tools)
        if args.exclude_domains:
            cfg.excluded_domains = [d.strip() for d in args.exclude_domains.split(",") if d.strip()]
        import os
        cfg.base_url = os.getenv("OPEN_FUSION_BASE_URL")
        cfg.api_key = os.getenv("OPEN_FUSION_API_KEY") or os.getenv("OPENROUTER_API_KEY")

        baseline = ModelSpec(args.baseline or cfg.judge.slug)
        client = ModelClient(fusion_depth=cfg.depth + 1, base_url=cfg.base_url, api_key=cfg.api_key)
        if args.grader == "llm":
            grader = LLMJudgeGrader(client, ModelSpec(args.grader_model))
        else:
            grader = RuleGrader()
        label = f"{args.preset or 'custom'} vs {baseline.slug}"
        report = asyncio.run(evaluate(tasks, cfg, baseline, grader,
                                      client=client, n_passes=args.passes, label=label))

    md = render_markdown(report)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    if args.json:
        sys.stdout.write(render_json(report) + "\n")
    else:
        sys.stdout.write(md + "\n")
    # the verdict line repeated on stderr for piping convenience
    print(f"\n[verdict] worth_it={report['verdict']['worth_it']} "
          f"rating={report['verdict']['rating']} lift={report['verdict']['lift']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
