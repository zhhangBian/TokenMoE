"""Training-only empirical survival with right censoring and unknown tails."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Duration:
    task: str
    category: str
    seconds: float
    censored: bool = False


class Survival:
    def __init__(self, observations, *, min_support=8):
        self.observations = list(observations)
        if any(o.seconds < 0 for o in self.observations):
            raise ValueError("Duration observations must be nonnegative")
        self.min_support = min_support

    def predict(self, category, *, elapsed=0.0):
        if elapsed < 0:
            raise ValueError("Elapsed time must be nonnegative")
        local = [o for o in self.observations if o.category == category]
        fallback = len(local) < self.min_support
        selected = self.observations if fallback else local
        at_risk, survival = len(selected), 1.0
        groups = defaultdict(lambda: [0, 0])
        for observation in selected:
            groups[observation.seconds][int(observation.censored)] += 1
        points = []
        for duration, (completed, censored) in sorted(groups.items()):
            if completed:
                survival *= 1 - completed / at_risk
            points.append((duration, survival))
            at_risk -= completed + censored
        before = 1.0
        for duration, probability in points:
            if duration <= elapsed:
                before = probability
        remaining = (
            [
                (duration - elapsed, 1 - probability / before)
                for duration, probability in points
                if duration > elapsed
            ]
            if before > 0
            else []
        )

        def quantile(target):
            return next((time for time, cdf in remaining if cdf >= target), None)

        # A completely unfamiliar history retains an unknown tail, never a false deadline.
        return {
            "support": len(selected),
            "category_support": len(local),
            "fallback": fallback,
            "elapsed": elapsed,
            "cdf": remaining,
            "p10": quantile(0.1),
            "p50": quantile(0.5),
            "p90": quantile(0.9),
            "unresolved_tail_probability": 1 - remaining[-1][1] if remaining else 1.0,
            "prefetch_supported": len(selected) >= self.min_support and before > 0,
            "training_tasks": sorted({o.task for o in selected}),
        }


def parallel_join_probability(predictions, horizon):
    probability = 1.0
    for prediction in predictions:
        completed = max((cdf for t, cdf in prediction["cdf"] if t <= horizon), default=0)
        probability *= completed
    return {"probability": probability, "assumption": "conditionally independent parallel tools"}
