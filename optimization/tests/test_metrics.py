import json

import pytest

from tokenmoe_opt.artifacts import manifest
from tokenmoe_opt.costs import fit
from tokenmoe_opt.metrics import token_metrics


def test_token_metrics_preserve_batch_resolution_and_separate_endpoint_tpot():
    fields = {"request_id": "r", "clock_domain_id": "clock"}
    events = [
        {**fields, "kind": "request_received", "source_at": 100},
        {
            **fields,
            "kind": "token_readiness",
            "source_at": 200,
            "processing_batch": 1,
            "token_indices": [0],
        },
        {
            **fields,
            "kind": "token_readiness",
            "source_at": 500,
            "processing_batch": 2,
            "token_indices": [1, 2],
        },
        {
            **fields,
            "kind": "request_finished",
            "source_at": 600,
            "num_output_tokens": 3,
            "status": "FINISHED_STOPPED",
        },
    ]
    result = token_metrics(events)[0]
    assert result["engine_ttft_seconds"] == 100 / 1e9
    assert result["endpoint_average_tpot_seconds"] == 150 / 1e9
    assert result["observed_itl_seconds"] is None
    assert result["network_delivery"] is None


def test_missing_initial_token_is_not_fabricated_as_ttft():
    result = token_metrics(
        [
            {
                "request_id": "r",
                "clock_domain_id": "clock",
                "kind": "token_readiness",
                "source_at": 500,
                "processing_batch": 2,
                "token_indices": [2],
            }
        ]
    )[0]
    assert result["engine_ttft_seconds"] is None
    assert not result["complete_token_coverage"]


@pytest.mark.parametrize("field", ["kv_cache_memory_bytes", "enforce_eager", "async_scheduling"])
def test_copy_cost_fitting_rejects_confounded_execution_profiles(tmp_path, field):
    keys = (
        "model tensor_parallel_size dtype max_model_len max_num_seqs "
        "max_num_batched_tokens kv_cache_memory_bytes enforce_eager async_scheduling "
        "enable_prefix_caching kernel_config"
    ).split()
    native = {"source_stable": True, "trials": [{"tokens": [1]}], "config": dict.fromkeys(keys, 1)}
    for name in ("native", "cache"):
        path = tmp_path / name
        path.mkdir()
        record = {
            **native,
            "config": {**native["config"], **({field: 2} if name == "cache" else {})},
        }
        (path / "result.json").write_text(json.dumps(record))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(manifest("input", {"model": "q"})))
    with pytest.raises(ValueError, match=field):
        fit(tmp_path / "native", tmp_path / "cache", inputs, tmp_path / "out", command=["test"])
    assert not (tmp_path / "out").exists()
