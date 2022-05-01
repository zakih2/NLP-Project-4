"""Helper functions.
"""

import random


def choice(population, weights):
    """Choose an element from a population given probabilistic weights.

    Parameters
    ----------
    population : list[Any]
      A population to choose an element from.
    weights : list[float | int]
      Weights given to the choice.

    Returns
    -------
    Any
      Returns an element of population.

    Raises
    ------
    TypeError
      If population and weights are not lists.
    """
    if not (isinstance(population, list) and isinstance(weights, list)):
        raise TypeError("Parameters for choice() should be type list.")
    s = sum(weights)
    weights = [w / s for w in weights]

    f = random.random()

    i = 0
    c = 0
    while c < f:
        c += weights[i]
        i += 1

    return population[i - 1]


def choices(population, weights, k=1):
    """Choose k elements from a population given probabilistic weights.

    Parameters
    ----------
    population : list[Any]
      A population to choose an element from.
    weights : list[float|int]
      Weights given to the choice.
    k : int
      Number of elements to choose. Defaults to 1.

    Returns
    -------
    list[Any]
      Returns an element of population.
    """
    return [choice(population, weights) for _ in range(k)]
