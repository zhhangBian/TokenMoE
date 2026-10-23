"""Bounded observed-prefix features and task-held-out survival predictions."""

import codecs
import io
import re

import numpy as np

from .arrival import Duration, Survival

FEATURE_NAMES = (
    "log_bytes",
    "log_chunks",
    "log_lines",
    "log_collected",
    "progress_fraction",
    "progress_present",
    "error_phase",
    "summary_phase",
    "log_output_age",
)


def command_family(command):
    if "compileall" in command:
        return "compile"
    if "runtests.py" in command:
        return "django"
    return "pytest" if "-m pytest" in command else "unknown"


class StreamState:
    def __init__(self):
        self.bytes = self.chunks = self.lines = self.collected = 0
        self.percent = None
        self.error = self.summary = False
        self.last_at = None
        self.partial = ""
        self.decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")("replace"), translate=True
        )

    def observe(self, data, available_at):
        if self.last_at is not None and available_at < self.last_at:
            raise ValueError("Stream availability time regressed")
        self.bytes += len(data)
        self.chunks += 1
        self.last_at = available_at
        text = self.partial + self.decoder.decode(data)
        lines = text.splitlines(keepends=True)
        self.partial = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.partial = lines.pop()[-4096:]
        for line in lines:
            line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
            self.lines += 1
            count = re.search(r"(?:collected|Found)\s+(\d{1,12})\s+(?:items?|tests?)", line)
            if count:
                self.collected = int(count[1])
            percent = re.findall(r"\[\s*(\d{1,3})%\s*\]", line)
            if percent and int(percent[-1]) <= 100:
                self.percent = int(percent[-1]) / 100
            self.error |= "Traceback (most recent call last)" in line or " ERROR " in line
            self.summary |= any(
                value in line
                for value in (
                    "short test summary info",
                    "FAILURES",
                    "Ran ",
                    " warnings summary ",
                )
            )

    def features(self, *, decision_at, elapsed):
        if elapsed < 0 or (self.last_at is not None and self.last_at > decision_at):
            raise ValueError("Features include an unavailable observation")
        age = elapsed if self.last_at is None else (decision_at - self.last_at) / 1e9
        return np.asarray(
            [
                np.log1p(self.bytes),
                np.log1p(self.chunks),
                np.log1p(self.lines),
                np.log1p(self.collected),
                self.percent or 0,
                self.percent is not None,
                self.error,
                self.summary,
                np.log1p(age),
            ],
            dtype=float,
        )


def probability(prediction, horizon):
    points = prediction["cdf"]
    if not points or (horizon > points[-1][0] and prediction["unresolved_tail_probability"] > 0):
        return None
    return max((cdf for time, cdf in points if time <= horizon), default=0)


class ProgressPredictor:
    def __init__(self, training, *, neighbors=8, min_support=4):
        if neighbors < min_support or min_support < 2:
            raise ValueError("Invalid stream prediction support")
        self.training = list(training)
        self.neighbors, self.min_support = neighbors, min_support
        self.tasks = {r["task"] for r in self.training}
        self.history = Survival(
            [Duration(r["task"], r["family"], r["seconds"], r["censored"]) for r in self.training],
            min_support=min_support,
        )

    def predict(self, method, *, task, family, elapsed, features):
        if task in self.tasks:
            raise ValueError("Prediction task was included in training")
        if (
            not np.isfinite(elapsed)
            or elapsed < 0
            or not np.isfinite(features).all()
            or len(features) != len(FEATURE_NAMES)
        ):
            raise ValueError("Invalid observed stream features")
        base = self.history.predict(family, elapsed=elapsed)
        base["source"] = "elapsed-history"
        if method == "elapsed-history":
            return base
        if method == "explicit-progress":
            if not features[5] or features[4] <= 0:
                return {**base, "source": "elapsed-history-no-explicit-progress"}
            remaining = elapsed * (1 - features[4]) / features[4]
            return {
                **base,
                "source": "explicit-progress-linear-point",
                "p50": remaining,
                "p10": None,
                "p90": None,
                "cdf": [(remaining, 1.0)],
                "unresolved_tail_probability": 0,
                "interval_reason": "Point extrapolation has no calibrated interval",
            }
        if method != "observed-stream":
            raise ValueError("Unknown stream prediction method")
        local = [r for r in self.training if r["family"] == family]
        pool = local if len(local) >= self.min_support else self.training
        if any(r["censored"] and r["seconds"] <= elapsed for r in pool):
            return {**base, "source": "elapsed-history-censored-prefix-unavailable"}
        active = [r for r in pool if r["seconds"] > elapsed and elapsed in r["features"]]
        if len(active) < self.min_support:
            return {**base, "source": "elapsed-history-insufficient-stream-support"}
        matrix = np.stack([r["features"][elapsed] for r in active])
        scales = np.maximum(matrix.std(axis=0), 0.25)
        distances = np.linalg.norm((matrix - features) / scales, axis=1)
        limit = np.sort(distances)[min(self.neighbors, len(active)) - 1]
        selected = [r for r, distance in zip(active, distances) if distance <= limit + 1e-12]
        if len({r["task"] for r in selected}) < 2:
            return {**base, "source": "elapsed-history-insufficient-task-diversity"}
        prediction = Survival(
            [Duration(r["task"], family, r["seconds"] - elapsed, r["censored"]) for r in selected],
            min_support=self.min_support,
        ).predict(family)
        return {
            **prediction,
            "source": "observed-prefix-neighbor-survival",
            "neighbor_jobs": [r["job_id"] for r in selected],
            "elapsed": elapsed,
        }
