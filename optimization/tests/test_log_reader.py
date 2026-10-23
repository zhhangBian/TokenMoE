import subprocess
import sys
from pathlib import Path

import pytest

from tokenmoe_opt.log_reader import parse_forecast


def test_reader_json_preserves_ordered_bounds_and_rejects_unusable_forecasts():
    assert parse_forecast('```json\n{"p10": 1, "remaining_seconds": 2, "p90": 4}\n```') == {
        "p10": 1,
        "remaining_seconds": 2,
        "p90": 4,
    }
    for invalid in [
        "no answer",
        '{"p10": true, "remaining_seconds": 2, "p90": 4}',
        '{"p10": 3, "remaining_seconds": 2, "p90": 4}',
        '{"p10": 1, "remaining_seconds": 2, "p90": NaN}',
    ]:
        with pytest.raises(ValueError):
            parse_forecast(invalid)


def test_reader_import_does_not_require_parquet_or_initialize_gpu():
    script = Path(__file__).resolve().parents[1] / "scripts/qualify_log_reader.py"
    code = """
import importlib.abc, runpy, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','vllm','pyarrow'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockHeavy())
runpy.run_path(sys.argv[1], run_name='qualification_import_only')
"""
    subprocess.run([sys.executable, "-c", code, str(script)], check=True)
