"""
"""

from metrics import fscore
from naive_bayes import NaiveBayesClassifier


def get_x_y(path: str) -> tuple:
    with open(path, "r") as f:
        lines = f.readlines()
    y = [int(l.split("\t")[-1]) for l in lines]
    x = [" ".join(l.split("\t")[0:-1]) for l in lines]

    return x, y


def main():

    x_train, y_train = get_x_y("./train.txt")
    x_test, y_test = get_x_y("./test.txt")

    clf = NaiveBayesClassifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    f = fscore(preds, y_test)

    print(f)
    print(a)


if __name__ == "__main__":
    main()
