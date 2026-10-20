"""Load the v2 golden cases."""
import json
from pathlib import Path

CASES = Path(__file__).resolve().parents[1] / "testdata" / "kiln-v2-cases.json"

def test_cases_load():
    data = json.loads(CASES.read_text())
    assert isinstance(data, (list, dict))
    assert data
