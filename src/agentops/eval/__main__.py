"""python -m agentops.eval writes eval-report.json and prints one line per case."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentops.eval.runner import run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentOps evaluation cases.")
    parser.add_argument(
        "--report",
        default="eval-report.json",
        help="Where to write the JSON report.",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(run_suite(Path(args.report)))
    for case in report["cases"]:
        flag = "pass" if case["success"] else "fail"
        print(
            f"{case['name']:<28} {flag:<4} status={case['status']} tool_calls={case['tool_calls']}"
        )
        if case["error"]:
            print(f"  {case['error']}")
    summary = report["summary"]
    print(
        f"cases={summary['total']} passed={summary['passed']} "
        f"success_rate={summary['success_rate']} tool_calls={summary['tool_calls']}"
    )
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
