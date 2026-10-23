"""Model-specific frequency priors and causal same-session history."""

import heapq
from collections import defaultdict

import numpy as np

POLICIES = (
    "global",
    "stage",
    "previous",
    "ewma",
    "recent",
    "previous+global",
    "previous+stage",
    "previous+tool",
    "hindsight_fixed_set",
)


def normalize(hist):
    total = hist.sum(axis=-1, keepdims=True)
    return np.divide(hist, total, out=np.full(hist.shape, 1 / hist.shape[-1]), where=total != 0)


def selected(distribution, budget):
    if not 0 < budget <= distribution.shape[-1]:
        raise ValueError("Expert budget must be within the model's expert count")
    # Historical screening used a stable ascending sort and the final K entries.
    return np.argsort(distribution, axis=-1, kind="stable")[..., -budget:]


def coverage(hist, distribution, budget):
    hits = np.take_along_axis(hist, selected(distribution, budget), axis=-1).sum()
    return int(hits), int(hist.sum())


class Priors:
    def __init__(self, train, *, min_support=8):
        if not train:
            raise ValueError("Priors require training observations")
        self.min_support = min_support
        self.global_hist = sum((s.hist for s in train), np.zeros_like(train[0].hist))
        self.tables, self.support = {}, defaultdict(int)
        for sample in train:
            for key in (("stage", sample.features["stage"]), ("tool", sample.features["tool_key"])):
                self.tables.setdefault(key, np.zeros_like(sample.hist))
                self.tables[key] += sample.hist
                self.support[key] += 1
        self.training_requests = sorted(s.request for s in train)

    def distribution(self, sample, kind):
        key = (kind, sample.features["stage"] if kind == "stage" else sample.features["tool_key"])
        support = self.support.get(key, 0)
        if kind == "global" or support < self.min_support:
            return normalize(self.global_hist), {
                "source": "global",
                "requested": kind,
                "key_support": support,
                "support": len(self.training_requests),
                "fallback": kind != "global",
            }
        table = self.tables[key]
        # A sparse/empty phase falls back per layer, even with sufficient request support.
        dist = normalize(table)
        empty = table.sum(axis=-1) == 0
        dist[empty] = normalize(self.global_hist)[empty]
        return dist, {"source": kind, "support": support, "fallback": bool(empty.any())}

    def artifact(self):
        return {
            "global_hist": self.global_hist.tolist(),
            "tables": [
                {
                    "kind": key[0],
                    "key": key[1],
                    "hist": value.tolist(),
                    "support": self.support[key],
                }
                for key, value in self.tables.items()
            ],
            "training_requests": self.training_requests,
            "min_support": self.min_support,
        }


class History:
    def __init__(self, *, alpha=0.5, ewma_alpha=0.5):
        if not 0 <= alpha <= 1 or not 0 <= ewma_alpha <= 1:
            raise ValueError("Mixture weights must be in [0, 1]")
        self.alpha, self.ewma_alpha = alpha, ewma_alpha
        self.previous, self.ewma = {}, {}
        self.ewma_support = {}
        self.pending = []
        self.stale_observations = 0

    def defer(self, sample):
        heapq.heappush(self.pending, (sample.observed, sample.decision, sample.request, sample))

    def _advance(self, decision_at):
        while self.pending and self.pending[0][0] <= decision_at:
            _, _, _, sample = heapq.heappop(self.pending)
            old = self.previous.get(sample.session)
            if old is not None and sample.decision <= old.decision:
                self.stale_observations += 1
            else:
                self.observe(sample)

    def predict(self, sample, prior, policy):
        self._advance(sample.decision)
        if policy not in POLICIES:
            raise ValueError(f"Unknown predictor: {policy}")
        kind = policy.split("+")[-1] if "+" in policy else policy
        if kind not in {"global", "stage", "tool"}:
            kind = "global"
        dist, provenance = prior.distribution(sample, kind)
        old = self.previous.get(sample.session)
        if old is not None and old.observed > sample.decision:
            raise ValueError("History observation is later than prediction decision")
        if policy == "hindsight_fixed_set":
            return normalize(sample.hist), {"source": "target-hindsight", "deployable": False}
        if old is None and policy not in {"global", "stage"}:
            global_dist, _ = prior.distribution(sample, "global")
            dist = (
                self.alpha * global_dist + (1 - self.alpha) * dist if "+" in policy else global_dist
            )
            provenance["cold_start"] = "global history; retain configured prior mixture"
        if old is not None:
            global_dist, _ = prior.distribution(sample, "global")
            observed = old.hist.sum(axis=-1, keepdims=True) > 0
            history_dist = np.where(observed, normalize(old.hist), global_dist)
            provenance["empty_history_distributions"] = int((~observed).sum())
            if policy == "previous":
                dist = history_dist
            elif policy == "recent":
                dist = np.where(observed, normalize(old.recent), global_dist)
            elif policy == "ewma":
                dist = np.where(
                    self.ewma_support[sample.session], self.ewma[sample.session], global_dist
                )
            elif "+" in policy:
                dist = self.alpha * history_dist + (1 - self.alpha) * dist
        return dist, {
            **provenance,
            "history_request": old.request if old else None,
            "history_observed_at": old.observed if old else None,
            "deployable": True,
        }

    def observe(self, sample):
        old = self.previous.get(sample.session)
        if old is not None and sample.decision <= old.decision:
            raise ValueError("History must advance chronologically")
        dist = normalize(sample.hist)
        observed = sample.hist.sum(axis=-1, keepdims=True) > 0
        if sample.session in self.ewma:
            old_dist = self.ewma[sample.session]
            update = self.ewma_alpha * dist + (1 - self.ewma_alpha) * old_dist
            self.ewma[sample.session] = np.where(
                observed, np.where(self.ewma_support[sample.session], update, dist), old_dist
            )
            self.ewma_support[sample.session] |= observed
        else:
            self.ewma[sample.session] = dist
            self.ewma_support[sample.session] = observed
        self.previous[sample.session] = sample
