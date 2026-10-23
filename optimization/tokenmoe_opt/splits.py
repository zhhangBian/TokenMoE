"""Task-grouped outer evaluation and training-only inner parameter selection."""

import numpy as np

from .artifacts import manifest


def grouped_splits(samples, identity):
    groups = sorted({s.task for s in samples})
    if len(groups) < 3:
        raise ValueError("At least three task groups are required for nested evaluation")
    return manifest(
        "split",
        identity,
        protocol="leave-one-task-out-exploratory",
        untouched_confirmation_groups=[],
        groups={g: sorted({s.session for s in samples if s.task == g}) for g in groups},
        folds=[
            {
                "test": [g],
                "train": [t for t in groups if t != g],
                "inner_validation": [t for t in groups if t != g],
            }
            for g in groups
        ],
    )


def tune_mixture(train, prior_factory, *, candidates=(0.0, 0.5, 1.0), budget=64):
    """Tune only on inner held-out tasks; history updates after each target ends."""
    from .predictors import coverage, normalize

    groups = sorted({s.task for s in train})
    if len(groups) < 2:
        raise ValueError("Inner tuning requires at least two training groups")
    scores = {alpha: [] for alpha in candidates}
    for held in groups:
        prior = prior_factory([s for s in train if s.task != held])
        previous = {}
        totals = {alpha: [0, 0] for alpha in candidates}
        for sample in sorted((s for s in train if s.task == held), key=lambda s: s.decision):
            old = previous.get(sample.session)
            if old is not None and old.observed > sample.decision:
                raise ValueError("Future request observation used for history")
            for alpha in candidates:
                pred = prior.distribution(sample, "global")[0]
                if old is not None:
                    pred = alpha * normalize(old.hist) + (1 - alpha) * pred
                numerator, denominator = coverage(sample.hist, pred, budget)
                totals[alpha][0] += numerator
                totals[alpha][1] += denominator
            previous[sample.session] = sample
        for alpha, (numerator, denominator) in totals.items():
            if denominator:
                scores[alpha].append(numerator / denominator)
    means = {
        alpha: float(np.mean(values)) if values else float("-inf")
        for alpha, values in scores.items()
    }
    return max(candidates, key=lambda alpha: (means[alpha], -alpha)), means
