"""Driving file to produce deliverables as needed.
"""

from sklearn.utils import shuffle

from metrics import fscore
from naive_bayes import NaiveBayesClassifier


def get_x_y(path: str) -> tuple:
    """Read an input file and return the document's text along with its label.

    Parameters
    ----------
    path : str
        Input file to read.

    Returns
    -------
    tuple[list[str], list[int]]
        Corresponding arrays of documents and labels.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    y = [int(l.split("\t")[-1]) for l in lines]
    x = [" ".join(l.split("\t")[0:-1]) for l in lines]

    return x, y


def main() -> None:
    """Main function.
    """

    x_train, y_train = get_x_y("./train.txt")
    x_test, y_test = get_x_y("./test.txt")

    clf = NaiveBayesClassifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    f = fscore(preds, y_test)

    print(f)


if __name__ == "__main__":
    main()
