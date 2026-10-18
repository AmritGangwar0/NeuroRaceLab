import importlib.util
import json
from pathlib import Path

RENDERER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "render_report.py"
)
SPEC = importlib.util.spec_from_file_location("render_report", RENDERER_PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
render_report = RENDERER.render_report
def test_render_report_creates_png(tmp_path):
    source, destination = tmp_path / "report.json", tmp_path / "report.png"
    report = {
        "config": {"run_id":"test-run","model":{"name":"test-controller"},"seeds":[42],"max_steps":100,"track_width_m":8.0,"purpose":"test"},
        "overall_summary": {"episodes":1,"completed_episodes":1,"completion_rate":1.0,"mean_progress":1.0,"mean_return":100.0,"mean_inference_latency_ms":0.1,"termination_reasons":{"lap_complete":1}},
        "track_summaries": {"oval":{"episodes":1,"completed_episodes":1,"completion_rate":1.0,"mean_progress":1.0,"mean_return":100.0}},
        "episodes": [{"track_family":"oval","seed":42,"completed":True,"progress":1.0,"episode_return":100.0,"inference_latency_mean_ms":0.1,"termination_reason":"lap_complete","mean_abs_lateral_error_m":0.2}],
    }
    source.write_text(json.dumps(report)); render_report(source,destination)
    assert destination.is_file()
    assert destination.stat().st_size > 10_000
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
