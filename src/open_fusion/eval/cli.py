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
    p.add_argument("--suite", choices=["sample", "draco", "longhorizon", "semiconductor", "fusion", "superiority", "all"],
                   default=None, help="Built-in task suite (overrides --tasks). Use 'superiority' for the benchmark proving fusion beats top single models.")
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
    p.add_argument("--ablation", action="store_true",
                   help="Run the factorial mechanism ablation (decomposes lift into "
                        "synthesis / diversity / structured-judge with paired bootstrap CIs).")
    p.add_argument("--superiority", action="store_true",
                   help="Run the superiority benchmark: prove Fusion beats every single model, "
                        "including the Oracle Best-of-N (per-task max over all solo models).")
    p.add_argument("--tie-margin", dest="tie_margin", type=float, default=0.5,
                   help="Score margin within which a head-to-head counts as a tie (default: 0.5 pts on 0-100 scale).")
    p.add_argument("--bootstrap-iters", dest="bootstrap_iters", type=int, default=2000,
                   help="Number of bootstrap iterations for significance tests (default: 2000).")
    p.add_argument("--md", default=None, help="Write the markdown report to this path.")
    p.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tasks = load_tasks(args.suite or args.tasks)

    # --- ablation: decompose the lift into its mechanisms (own report shape) -----
    if args.ablation:
        from .ablation import render_ablation_markdown, run_ablation, run_ablation_demo
        if args.demo:
            report = run_ablation_demo(tasks)
        else:
            import os
            base = load_preset(args.preset or "quality") if not args.panel else FusionConfig(
                panel=[ModelSpec(s.strip()) for s in args.panel.split(",") if s.strip()],
                judge=ModelSpec(args.judge or "openai/gpt-5.5"))
            base.tools_enabled = bool(args.tools)
            if env_base_url := os.getenv("OPEN_FUSION_BASE_URL"):
                base.base_url = env_base_url
            env_api_key = os.getenv("OPEN_FUSION_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if env_api_key:
                base.api_key = env_api_key
            # ablation needs DISTINCT models; the panel doubles as the model pool.
            models = base.panel if len(base.panel) >= 2 else [base.panel[0], base.judge]
            panel_size = len(base.panel)
            executor_workers = max(panel_size * 2, 8) if panel_size >= 4 else None
            client = ModelClient(fusion_depth=base.depth + 1, base_url=base.base_url,
                                 api_key=base.api_key, executor_workers=executor_workers)
            grader = (LLMJudgeGrader(client, ModelSpec(args.grader_model))
                      if args.grader == "llm" else RuleGrader())
            report = asyncio.run(run_ablation(tasks, models, base.judge, grader,
                                              client=client, base=base, n_passes=args.passes))
        md = render_ablation_markdown(report)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as f:
                f.write(md + "\n")
        if args.json:
            sys.stdout.write(render_json(report) + "\n")
        else:
            sys.stdout.write(md + "\n")
        return 0

    # --- superiority benchmark: prove fusion beats best single model + oracle --
    if args.superiority:
        from .superiority import (render_superiority_markdown,
                                  run_superiority_benchmark, run_superiority_demo)
        if args.demo:
            report = run_superiority_demo(tasks, seed=42)
        else:
            import os
            base = load_preset(args.preset or "quality") if not args.panel else FusionConfig(
                panel=[ModelSpec(s.strip()) for s in args.panel.split(",") if s.strip()],
                judge=ModelSpec(args.judge or "openai/gpt-5.5"))
            base.tools_enabled = bool(args.tools)
            if env_base_url := os.getenv("OPEN_FUSION_BASE_URL"):
                base.base_url = env_base_url
            env_api_key = os.getenv("OPEN_FUSION_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if env_api_key:
                base.api_key = env_api_key
            models = base.panel if len(base.panel) >= 2 else [base.panel[0], base.judge]
            panel_size = len(base.panel)
            executor_workers = max(panel_size * 2, 8) if panel_size >= 4 else None
            client = ModelClient(fusion_depth=base.depth + 1, base_url=base.base_url,
                                 api_key=base.api_key, executor_workers=executor_workers)
            grader = (LLMJudgeGrader(client, ModelSpec(args.grader_model))
                      if args.grader == "llm" else RuleGrader())
            report = asyncio.run(run_superiority_benchmark(
                tasks, models, base.judge, grader,
                client=client, base=base, n_passes=args.passes,
                bootstrap_iters=args.bootstrap_iters, tie_margin=args.tie_margin,
                panel_size=panel_size, seed=42))
        md = render_superiority_markdown(report)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as f:
                f.write(md + "\n")
        if args.json:
            sys.stdout.write(render_json(report) + "\n")
        else:
            sys.stdout.write(md + "\n")
        return 0

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
        if env_base_url := os.getenv("OPEN_FUSION_BASE_URL"):
            cfg.base_url = env_base_url
        env_api_key = os.getenv("OPEN_FUSION_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if env_api_key:
            cfg.api_key = env_api_key

        baseline = ModelSpec(args.baseline or cfg.judge.slug)
        panel_size = len(cfg.panel)
        executor_workers = max(panel_size * 2, 8) if panel_size >= 4 else None
        client = ModelClient(fusion_depth=cfg.depth + 1, base_url=cfg.base_url, api_key=cfg.api_key,
                             executor_workers=executor_workers)
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
