"""The generator writes the same text on a second run, and the structs round-trip."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_point_round_trip(tmp_path: Path):
    out = tmp_path / "point.py"
    subprocess.check_call(
        [sys.executable, "-m", "kiln.codegen", str(ROOT / "testdata/schema/point.json"), "-o", str(out)],
        cwd=ROOT,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
    )
    again = tmp_path / "point2.py"
    subprocess.check_call(
        [sys.executable, "-m", "kiln.codegen", str(ROOT / "testdata/schema/point.json"), "-o", str(again)],
        cwd=ROOT,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
    )
    assert out.read_text() == again.read_text()
    import importlib.util

    spec = importlib.util.spec_from_file_location("point_gen", out)
    module = importlib.util.module_from_spec(spec)
    sys.modules["point_gen"] = module
    spec.loader.exec_module(module)
    point = module.Point(x=3, y=-1, label="n")
    assert module.Point.decode(point.encode()) == point
