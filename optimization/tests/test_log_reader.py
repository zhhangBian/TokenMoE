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
