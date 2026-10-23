"""Task-grouped outer evaluation and training-only inner parameter selection."""

import numpy as np

from .artifacts import manifest


def grouped_splits(samples, identity, *, held_out=None):
    groups = sorted({s.task for s in samples})
    if len(groups) < 3:
        raise ValueError("At least three task groups are required for nested evaluation")
    tests = [[g] for g in groups]
    protocol = "leave-one-task-out-exploratory"
    if held_out is not None:
        held_out = sorted(set(held_out))
        if not held_out or not set(held_out) <= set(groups) or len(groups) - len(held_out) < 2:
            raise ValueError("Holdout requires known task groups and at least two training groups")
        tests = [held_out]
        protocol = "frozen-multiple-task-holdout-exploratory"
    return manifest(
        "split",
        identity,
        protocol=protocol,
        untouched_confirmation_groups=[],
        groups={g: sorted({s.session for s in samples if s.task == g}) for g in groups},
        folds=[
            {
                "test": test,
                "train": [t for t in groups if t not in test],
                "inner_validation": [t for t in groups if t not in test],
            }
            for test in tests
        ],
    )


def tune_mixture(train, prior_factory, *, candidates=(0.0, 0.5, 1.0), budget=64):
    """Tune only on inner held-out tasks; history updates after each target ends."""
    from .predictors import History, coverage

    groups = sorted({s.task for s in train})
    if len(groups) < 2:
        raise ValueError("Inner tuning requires at least two training groups")
    scores = {alpha: [] for alpha in candidates}
    for held in groups:
        prior = prior_factory([s for s in train if s.task != held])
        histories = {alpha: History(alpha=alpha) for alpha in candidates}
        totals = {alpha: [0, 0] for alpha in candidates}
        for sample in sorted((s for s in train if s.task == held), key=lambda s: s.decision):
            for alpha in candidates:
                history = histories[alpha]
                pred = history.predict(sample, prior, "previous+global")[0]
                numerator, denominator = coverage(sample.hist, pred, budget)
                totals[alpha][0] += numerator
                totals[alpha][1] += denominator
                history.defer(sample)
        for alpha, (numerator, denominator) in totals.items():
            if denominator:
                scores[alpha].append(numerator / denominator)
    means = {alpha: float(np.mean(values)) if values else None for alpha, values in scores.items()}
    supported = [alpha for alpha in candidates if means[alpha] is not None]
    if not supported:
        return (0.5 if 0.5 in candidates else candidates[0]), means
    return max(supported, key=lambda alpha: (means[alpha], -alpha)), means
