import numpy as np
import pytest

from tokenmoe_opt.tool_progress import FEATURE_NAMES, ProgressPredictor, StreamState, probability


def test_prefix_state_handles_split_utf8_and_never_infers_completion_from_percent():
    state = StreamState()
    text = "collected 10 items\ntest_λ PASSED [100%]\n".encode()
    cut = text.index("λ".encode()) + 1
    state.observe(text[:cut], 1)
    before = state.features(decision_at=1, elapsed=1)
    assert before[3] == np.log1p(10) and before[5] == 0
    state.observe(text[cut:], 5)
    assert state.percent == 1 and state.lines == 2
    with pytest.raises(ValueError, match="unavailable"):
        state.features(decision_at=4, elapsed=4)
    state.observe(b"x" * 10000, 6)
    assert len(state.partial) <= 4096
    newline = StreamState()
    newline.observe(b"one\r", 1)
    newline.observe(b"\ntwo\r\n", 2)
    assert newline.lines == 2


def training():
    zero = np.zeros(len(FEATURE_NAMES))
    return [
        {
            "task": f"train-{i}",
            "job_id": f"job-{i}",
            "family": "pytest",
            "seconds": 2 + i,
            "censored": False,
            "features": {0.0: zero.copy(), 1.0: zero.copy()},
        }
        for i in range(10)
    ]


def test_equal_prefixes_keep_all_neighbor_ties_and_training_groups_stay_excluded():
    rows = training()
    model = ProgressPredictor(rows)
    kwargs = dict(
        task="held-out", family="pytest", elapsed=0.0, features=np.zeros(len(FEATURE_NAMES))
    )
    predicted = model.predict("observed-stream", **kwargs)
    assert len(predicted["neighbor_jobs"]) == 10
    assert "held-out" not in predicted["training_tasks"]
    with pytest.raises(ValueError, match="training"):
        model.predict("observed-stream", **{**kwargs, "task": "train-0"})
    assert probability(predicted, 1) == 0


def test_censored_prefixes_cannot_be_silently_discarded_after_observation_ends():
    rows = training()
    rows[0].update(censored=True, seconds=0.5)
    result = ProgressPredictor(rows).predict(
        "observed-stream",
        task="held-out",
        family="pytest",
        elapsed=1.0,
        features=np.zeros(len(FEATURE_NAMES)),
    )
    assert result["source"] == "elapsed-history-censored-prefix-unavailable"
    unresolved = {"cdf": [(1, 0.0)], "unresolved_tail_probability": 1}
    assert probability(unresolved, 2) is None
