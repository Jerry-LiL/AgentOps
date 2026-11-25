from pathlib import Path

from agentops.eval.runner import run_suite


async def test_eval_suite_passes(tmp_path):
    report_path = tmp_path / "eval-report.json"
    report = await run_suite(report_path, root=tmp_path / "runs")
    failures = [case for case in report["cases"] if not case["success"]]
    assert failures == []
    assert report["summary"]["passed"] == report["summary"]["total"]
    assert report["summary"]["total"] == 6
    assert report["summary"]["tool_calls"] >= 6
    saved = Path(report_path).read_text()
    assert "success_rate" in saved
    assert "tool_calls" in saved
