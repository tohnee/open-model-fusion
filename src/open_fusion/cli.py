"""
open_fusion.cli - the `open-fusion` console script.

Contract: the final answer goes to STDOUT; everything else (telemetry, the optional
structured analysis, errors) goes to STDERR. That lets you do:
    open-fusion "..." > answer.md
and capture only the answer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import from_cli
from .orchestrator import fuse
from .schema import FusionStatus


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="open-fusion",
        description="Multi-model deliberation: fan a prompt to a panel, judge, synthesize.")
    p.add_argument("question", help="The question/prompt to deliberate on.")
    p.add_argument("--preset", choices=["quality", "budget"], default=None,
                   help="Built-in panel (default: quality).")
    p.add_argument("--panel", default=None,
                   help="Comma-separated model slugs (overrides --preset).")
    p.add_argument("--judge", default=None, help="Judge model slug.")
    p.add_argument("--tools", action="store_true", help="Enable grounded panels (web/exec).")
    p.add_argument("--exclude-domains", dest="exclude_domains", default=None,
                   help="Comma-separated domains to exclude from search/fetch.")
    p.add_argument("--max-in-flight", dest="max_in_flight", type=int, default=None,
                   help="Max concurrent panel calls.")
    p.add_argument("--fast-majority-k", dest="fast_majority_k", type=int, default=None,
                   help="Proceed as soon as K panels succeed (cancel the rest).")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON to stdout.")
    p.add_argument("--show-analysis", action="store_true",
                   help="Print the structured analysis to stderr.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = from_cli(args)

    result = asyncio.run(fuse(args.question, cfg))

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(result.text + ("\n" if result.text else ""))

    # diagnostics on stderr
    print(f"\n[status] {result.status.value if isinstance(result.status, FusionStatus) else result.status}",
          file=sys.stderr)
    print(f"[telemetry] {json.dumps(result.telemetry)}", file=sys.stderr)
    if result.error:
        print(f"[error] {result.error}", file=sys.stderr)
    if args.show_analysis and result.analysis is not None:
        print("[analysis] " + json.dumps(result.analysis.to_dict(), indent=2), file=sys.stderr)

    return 0 if result.status == FusionStatus.OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
