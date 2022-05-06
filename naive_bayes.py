"""Multinomial Naive Bayes Classifier for binary text classification tasks.

TODO:
- Implement binary mode for the classifier, i.e., T5. One way to do this is to
    add an additional parameter to the santiize function. When this parameter
    is properly set, the sanitize function can remove non-unique tokens. Then,
    in both the NaiveBayesClassifier.fit and the NaiveBayesClassifier.predict
    methods, change the call to sanitize to include the new parameter. When the
    NaiveBayesClassifier is instantiated with the mode="binary" flag, the new
    parameter should be set such that the sanitize function will remove
    duplicates from the string.
- Use log space for the prediction calculation. To do this, you'll need to
    modify the NaiveBayesClassifier.predict method.
"""


from __future__ import annotations
from collections import Counter
import math
import re


def sanitize(text: str) -> str:
    """Santiize the text to prepare it for learning.

    Parameters
    ----------
    text : str
        Text to be sanitized.

    Returns
    -------
    str
        Sanitized text.
    """
    text = text.lower()
    text = re.sub("[-\n]+", " ", text)
    text = re.sub("[?!;\.]+", " ", text)
    text = re.sub("[^a-zA-Z\. ]+", "", text)
    text = re.sub("[\s]{2,}", " ", text)
    text = text.strip()

    return text


def p_w_cls(n_k: int, delta: float, n_c: int, n_v: int) -> float:
    """Return the probability of a word occuring given a class.

    Parameters
    ----------
    n_k : int
        Number of occurences of the word in documents belonging to the class.
    delta : float
        The smoothing parameter.
    n_c : int
        Number of tokens in documents belonging to the class.
    n_v : int
        Size of the vocabulary.

    Returns
    -------
    float
        The probability of a word occuring given a class
    """
    return (n_k + delta) / (n_c + delta * n_v)


class NaiveBayesClassifier:
    """Multinomial Naive Bayes Classifier for binary text classification tasks.

    Attributes
    ----------
    mode : str
        The mode of feature extraction for the classifier, one of "count"
        or "binary". If "count", the classifier will count the number of times
        each word occurs in the training data. If "binary", the classifier
        considers words that occur in the training data many times as only
        occuring once.
    delta : float
        A smoothing parameter.
    p_neg : float
        The probability of a document belonging to the negative class based
        soley upon the distribution of the training data.
    p_pos : float
        The probability of a document belonging to the positive class based
        soley upon the distribution of the training data.
    p_w_neg : dict[str, float]
        The probability of a word occuring given the negative class.
    p_w_pos : dict[str, float]
        The probability of a word occuring given the positive class.

    Usage
    -----
    >>> x_train = ["I love this movie", "This is a great movie", "I hate this movie"]
    >>> y_train = [1, 1, 0]
    >>> x_test = ["I am not sure if I like this movie", "I hate this movie a lot"]
    >>> clf = NaiveBayesClassifier()
    >>> clf.fit(x_train, y_train)
    >>> clf.predict(x_test)
    ... [1, 0]
    """

    def __init__(self, mode: str = "count", delta: float = 1) -> None:
        """Create a classifier.

        Parameters
        ----------
        mode : str, optional
            The mode of feature extraction for the classifier, one of "count"
            or "binary", by default "count". If "count", the classifier will
            count the number of times each word occurs in the training data.
            If "binary", the classifier considers words that occur in the
            training data many times as only occuring once.
        delta : float, optional
            A smoothing parameter., by default 1
        """
        self.mode = mode
        self.delta = delta
        self.p_neg = None
        self.p_pos = None
        self.p_w_neg = None
        self.p_w_pos = None

    def fit(self, x: list, y: list) -> NaiveBayesClassifier:
        """Fit the classifier on training data.

        Parameters
        ----------
        x : list[str]
            A list of textual documents. Documents will undergo sanitization
            before learning commences.
        y : list[int]
            A corresponding list of labels, one for each document.

        Returns
        -------
        NaiveBayesClassifier
            The fitted classifier.
        """
        # Determine the probabilities of each class occurring from distribution
        class_dist = Counter(y)
        self.p_neg = class_dist[0] / len(y)
        self.p_pos = class_dist[1] / len(y)

        # Preprocess the training text
        x = [sanitize(x_i) for x_i in x]

        # Extract the vocabulary
        vocab = set([word for x_i in x for word in x_i.split()])
        n_v = len(vocab)

        # Extract the vocabulary and counts for the negative class
        x_neg = [x_i for x_i, y_i in zip(x, y) if y_i == 0]
        counter_neg = Counter([word for x_i in x_neg for word in x_i.split()])
        n_n = len(counter_neg)

        # Extract the vocabulary and counts for the positive class
        x_pos = [x_i for x_i, y_i in zip(x, y) if y_i == 1]
        counter_pos = Counter([word for x_i in x_pos for word in x_i.split()])
        n_p = len(counter_pos)

        # Determine the probability of a word given each class
        self.p_w_neg = {w: p_w_cls(counter_neg[w], self.delta, n_n, n_v) for w in vocab}
        self.p_w_pos = {w: p_w_cls(counter_pos[w], self.delta, n_p, n_v) for w in vocab}

        return self

    def predict(self, x: list) -> list:
        """Predict the class membership for novel testing data.

        Parameters
        ----------
        x : list
            A list of textual documents. Documents will undergo sanitization
            before learning commences.

        Returns
        -------
        list
            A corresponding list of prediction labels, one for each document.

        Raises
        ------
        ValueError
            If the classifier was not fitted prior to calling predict.
        """
        if any(x is None for x in (self.p_neg, self.p_pos, self.p_w_neg, self.p_w_pos)):
            raise ValueError("The classifier has not been fitted yet.")

        x = [sanitize(x_i) for x_i in x]

        y_hat = []
        for x_i in x:
            neg = self.p_neg * math.prod(
                self.p_w_neg[word] for word in x_i.split() if word in self.p_w_neg
            )
            pos = self.p_neg * math.prod(
                self.p_w_pos[word] for word in x_i.split() if word in self.p_w_pos
            )
            y_hat.append(1 if pos > neg else 0)

        return y_hat
