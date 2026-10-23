import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tokenmoe_opt.artifacts import Output, manifest, read_manifest
from tokenmoe_opt.data import Sample, phase_histograms
from tokenmoe_opt.features import command_categories, request_features
from tokenmoe_opt.predictors import POLICIES, History, Priors, coverage, normalize, selected
from tokenmoe_opt.protocol import local_protocol
from tokenmoe_opt.replay import CopyCost, replay, token_microbatches
from tokenmoe_opt.splits import grouped_splits, tune_mixture


def sample(task, session, *, decision=10, observed=20, expert=0, first=True):
    hist = np.zeros((3, 2, 4), dtype=np.int64)
    hist[..., expert] = 8
    return Sample(
        task,
        session,
        f"{session}-{decision}",
        decision,
        observed,
        hist,
        hist.copy(),
        {"stage": "first" if first else "subsequent", "tool_key": (), "first_request": first},
        Path("/unused"),
        {},
    )


def test_cpu_entrypoints_do_not_import_gpu_stack():
    script = """
import importlib.abc, sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'triton'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockGPU())
import tokenmoe_opt.analysis, tokenmoe_opt.cli, tokenmoe_opt.protocol, tokenmoe_opt.replay
import tokenmoe_opt.arrival_eval, tokenmoe_opt.planner, tokenmoe_opt.resources
import tokenmoe_opt.session, tokenmoe_opt.validation
import tokenmoe_opt.control_run, tokenmoe_opt.controller, tokenmoe_opt.costs, tokenmoe_opt.metrics
import tokenmoe_opt.agent_metrics
import tokenmoe_opt.replay_cache, tokenmoe_opt.timing, tokenmoe_opt.calibration
import tokenmoe_opt.agent_compare
import tokenmoe_opt.tool_corpus
import tokenmoe_opt.tool_progress, tokenmoe_opt.tool_progress_eval
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_phase_targets_use_actual_positions_and_empty_targets():
    # Prefix positions 0..2 and position 5 are absent, final sampled token is absent.
    positions = np.array([3, 4, 6, 7, 8], dtype=np.int32)
    routes = np.array([[[0, 1], [2, 3]]] * 5)
    hist, recent = phase_histograms(positions, routes, 7, 4)
    assert hist.sum(axis=(1, 2)).tolist() == [12, 8, 8]
    assert np.array_equal(hist, recent)
    empty, _ = phase_histograms(positions[:0], routes[:0], 7, 4)
    assert not empty.any()
    assert np.allclose(normalize(empty).sum(axis=-1), 1)
    assert coverage(empty, normalize(empty), 2) == (0, 0)


def test_first32_target_counts_positions_without_interpolating_gaps():
    positions = np.arange(10, 90, 2)
    routes = np.zeros((40, 2, 1), dtype=int)
    hist, recent = phase_histograms(positions, routes, 10, 4)
    assert hist.sum(axis=(1, 2)).tolist() == [0, 80, 64]
    assert recent[1].sum() == 64


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat file; python -m pytest tests", ("inspect", "test")),
        ("echo 'pytest fail'", ("unknown",)),
        ("rg test file", ("inspect",)),
        ("A=1 pytest -q && make", ("build_or_install", "test")),
        ("printf x > file", ("edit", "unknown")),
        ("python - <<'PY'\nprint('pytest')\nPY", ("unknown",)),
    ],
)
def test_shell_labels_preserve_compound_commands_without_labeling_payload(command, expected):
    assert command_categories(command) == expected


def test_historical_receipt_proxy_does_not_use_finalized_issuance(tmp_path):
    (tmp_path / "args").write_text("pytest")
    tools = {
        "t": {
            "tool_args_ref": "args",
            "issued_at": 1,
            "parsed_at": 5,
            "finished_at": 9,
            "exit_status": 0,
        }
    }
    req = {"request_created_at": 10, "prev_tool_call_ids": ["t"]}
    features = request_features(req, tools, tmp_path, first_request=False)
    assert features["availability"][0]["available_at"] == 10
    assert features["availability"][0]["source_observed_at"] == 5
    assert not features["first_request"] and not features["no_tool"]
    features = request_features(
        {**req, "prev_tool_call_ids": []}, tools, tmp_path, first_request=False
    )
    assert features["no_tool"] and not features["first_request"]
    with pytest.raises(ValueError, match="Unavailable"):
        request_features({**req, "request_created_at": 6}, tools, tmp_path, first_request=False)


def test_splits_keep_pilot_recovery_and_concurrency_copies_together():
    samples = [
        sample(task, f"{task}-{replica}")
        for task in ("a", "b", "c", "d")
        for replica in ("N1", "N4", "pilot", "recovery")
    ]
    splits = grouped_splits(samples, {"model": "qwen"})
    for fold in splits["folds"]:
        assert not set(fold["test"]) & set(fold["train"])
        assert set(fold["inner_validation"]) <= set(fold["train"])
    joint = grouped_splits(samples, {"model": "qwen"}, held_out=["c", "d"])
    assert joint["folds"] == [
        {"test": ["c", "d"], "train": ["a", "b"], "inner_validation": ["a", "b"]}
    ]
    assert joint["groups"] == splits["groups"]
    for invalid in ([], ["unknown"], ["b", "c", "d"]):
        with pytest.raises(ValueError, match="Holdout"):
            grouped_splits(samples, {"model": "qwen"}, held_out=invalid)
    train = [s for s in samples if s.task != "d"]
    fits = []

    def factory(items):
        fits.append({s.task for s in items})
        return Priors(items)

    tune_mixture(train, factory, budget=2)
    assert all("d" not in fitted and len(fitted) == 2 for fitted in fits)


def test_predictors_cannot_read_target_routes_or_future_observations():
    prior = Priors([sample("training", "train")])
    history = History()
    old = sample("held", "held", decision=1, observed=9, expert=1)
    history.observe(old)
    target = sample("held", "held", decision=10, expert=2, first=False)
    changed = copy.deepcopy(target)
    changed.hist = np.roll(changed.hist, 1, axis=-1)
    for policy in POLICIES[:-1]:
        predicted, provenance = history.predict(target, prior, policy)
        assert np.array_equal(predicted, history.predict(changed, prior, policy)[0])
        assert np.allclose(predicted.sum(axis=-1), 1)
        assert provenance["history_observed_at"] <= target.decision
    with pytest.raises(ValueError, match="later"):
        history.predict(sample("held", "held", decision=8), prior, "previous")
    assert selected(np.ones(4), 2).tolist() == [2, 3]


def test_sparse_tool_backoff_reports_training_support():
    prior = Priors([sample("training", "train")], min_support=8)
    predicted, provenance = prior.distribution(sample("held", "held"), "tool")
    assert provenance["fallback"] and provenance["key_support"] == 1
    assert predicted[..., 0].min() == 1


def test_overlapping_partial_requests_only_update_history_after_observation():
    prior = Priors([sample("train", "train", expert=0)])
    history = History()
    late = sample("held", "s", decision=1, observed=30, expert=1)
    retry = sample("held", "s", decision=10, observed=20, expert=2)
    history.defer(late)
    prediction, provenance = history.predict(retry, prior, "previous")
    assert provenance["history_request"] is None
    assert prediction[..., 0].min() == 1
    history.defer(retry)
    following = sample("held", "s", decision=25, observed=40, expert=3)
    assert history.predict(following, prior, "previous")[1]["history_request"] == retry.request
    following.decision = 35
    assert history.predict(following, prior, "previous")[1]["history_request"] == retry.request
    assert history.stale_observations == 1


def test_empty_inner_targets_keep_serializable_null_scores():
    samples = [sample(task, task) for task in ("a", "b", "c")]
    for observation in samples:
        observation.hist.fill(0)
    alpha, scores = tune_mixture(samples, Priors, budget=2)
    assert alpha == 0.5 and all(value is None for value in scores.values())
    json.dumps(scores, allow_nan=False)


def test_empty_phase_does_not_dilute_ewma_and_previous_uses_explicit_backoff():
    prior = Priors([sample("train", "train", expert=0)])
    history = History()
    history.observe(sample("held", "s", decision=1, observed=2, expert=1))
    partial = sample("held", "s", decision=3, observed=4, expert=2)
    partial.hist[1:] = 0
    partial.recent[1:] = 0
    history.observe(partial)
    target = sample("held", "s", decision=5)
    assert history.predict(target, prior, "ewma")[0][1, 0, 1] == 1
    assert history.predict(target, prior, "previous")[0][1, 0, 0] == 1


def test_history_cold_start_mixes_global_history_with_available_stage_prior():
    first = sample("training", "first", expert=0)
    subsequent = sample("training", "subsequent", expert=1, first=False)
    prior = Priors([first, subsequent], min_support=1)
    target = sample("held", "held")
    predicted, provenance = History().predict(target, prior, "previous+stage")
    expected = 0.5 * normalize(prior.global_hist) + 0.5 * normalize(first.hist)
    assert np.array_equal(predicted, expected)
    assert provenance["cold_start"].startswith("global history")


def test_output_reuse_is_rejected_and_failed_runs_keep_evidence(tmp_path):
    path = tmp_path / "run"
    out = Output(path, kind="result", identity={"model": "qwen"}, command=["test"])
    out.fail(ValueError("bad input"))
    assert read_manifest(path / "manifest.json", "result")["status"] == "failed"
    with pytest.raises(FileExistsError):
        Output(path, kind="result", identity={"model": "qwen"}, command=["test"])
    (tmp_path / "invalid.json").write_text(
        json.dumps({**manifest("input", {"m": 1}), "schema_version": 100})
    )
    with pytest.raises(ValueError, match="Unsupported"):
        read_manifest(tmp_path / "invalid.json", "input")


def test_protocol_requires_disjoint_confirmation_and_preserves_agent_budget():
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096}
    with pytest.raises(ValueError, match="required"):
        local_protocol({"model": "qwen"}, ["explored"], [], sampling=sampling)
    with pytest.raises(ValueError, match="overlap"):
        local_protocol({"model": "qwen"}, ["explored"], ["explored"], sampling=sampling)
    protocol = local_protocol({"model": "qwen"}, ["explored"], ["untouched"], sampling=sampling)
    assert protocol["untouched_confirmation_tasks"] == ["untouched"]
    assert protocol["sampling"]["max_tokens"] == 4096
    assert protocol["sampling"]["temperature"] == 0.6
    assert protocol["latency_targets"]["tools_to_complete_response_p95_seconds"] == 30
    assert protocol["measured_repetitions"] >= 3
    assert protocol["devices"] == [2, 3]
    assert all(status == "outstanding" for status in protocol["milestones"].values())
    expert = local_protocol(
        {"model": "qwen"},
        ["explored"],
        ["untouched"],
        sampling=sampling,
        prefill_execution="expert-major",
    )
    assert expert["protocol_id"] != protocol["protocol_id"]
    assert expert["sampling"] == protocol["sampling"]
    assert expert["latency_targets"] == protocol["latency_targets"]
    assert expert["mechanism"]["cache_scratch_bytes_per_rank"] == 192 * 1024**2
    assert "previous+tool" not in expert["policies"]


def test_overflow_partitions_preserve_tokens_topk_and_shared_loads():
    routes = [[0, 1], [1, 2], [2, 3], [0, 3]]
    batches = token_microbatches(routes, 3)
    assert batches == [(0, 2, (0, 1, 2)), (2, 4, (0, 2, 3))]
    assert [i for a, b, _ in batches for i in range(a, b)] == list(range(4))
    cost = CopyCost(1024, 1024)
    result = replay([{"time_seconds": 0, "layer": 0, "tokens": routes}], capacity=3, cost=cost)
    assert result["bytes"] == 4096
    assert result["unique_demand_hits"] == 2
    assert result["evidence"].endswith("diagnostic")
    with pytest.raises(ValueError, match="exceeds"):
        token_microbatches(routes, 1)
    with pytest.raises(ValueError, match="recompute"):
        replay([{"phase": "recompute"}], capacity=3, cost=cost)
