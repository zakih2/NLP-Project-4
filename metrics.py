"""Evaluation metrics for classification systems.
"""

# Values that inicate a positive or negative classification of an example.
true = 1
false = 0


def fscore(
    predictions: list, labels: list, beta: float = 1, zero_division: int = 0
) -> float:
    """Compute the fscore for a binary classification system.

    Parameters
    ----------
    predictions : list
        Predictions made by the classifier, consisting of values of
            `true` or `false`.
    labels : list
        Ground truth labels for the examples, consisting of values of
            `true` or `false`.
    beta : float, optional
        Value such that recall is beta-times more weighted than precision,
            by default 1.
    zero_division : int, optional
        Value to return when all predictions and all ground truth labels are
            negative, either 0 or 1, by default 0.

    Returns
    -------
    float
        The fscore.

    Raises
    ------
    ValueError
        If the number of predictions and labels is not equal.
    ValueError
        If zero_division is not 0 or 1.
    ValueError
        If an element of predictions or labels is not `true` or `false`.
    """

    if len(predictions) != len(labels):
        raise ValueError(
            "Length of predictions and labels must be equal, but got "
            f"{len(predictions)} predictions and {len(labels)} labels."
        )
    if zero_division not in {0, 1}:
        raise ValueError(f"zero_division must be 0 or 1, but got {zero_division}.")

    tp, fp, fn, tn = 0, 0, 0, 0

    for p, l in zip(predictions, labels):
        if p == true and l == true:
            tp += 1
        elif p == true and l == false:
            fp += 1
        elif p == false and l == true:
            fn += 1
        elif p == false and l == false:
            tn += 1
        else:
            raise ValueError(
                f"Unexpected value(s) in predictions or labels: {p, l}. "
                f"Expected either {true} or {false}."
            )

    if tp == 0:
        return zero_division

    f = ((1 + beta**2) * tp) / ((1 + beta**2) * tp + beta**2 * fn + fp)

    return f
