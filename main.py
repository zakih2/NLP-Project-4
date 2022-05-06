#!/home/hpc/kurlanl1/courses/CSC-427/project4/NLP-Project-4/env/bin/python -u

#SBATCH --chdir=/home/hpc/kurlanl1/courses/CSC-427/project4/NLP-Project-4
#SBATCH --output=/home/hpc/kurlanl1/courses/CSC-427/project4/NLP-Project-4/job.name.%A.out
#SBATCH --constraint=skylake|broadwell
#SBATCH --job-name=NLP
#SBATCH --partition=long

"""Driving file to produce deliverables as needed.
"""

import sys

sys.path.insert(0, "/home/hpc/kurlanl1/courses/CSC-427/project4/NLP-Project-4")

from argparse import ArgumentParser
from pathlib import Path

from metrics import fscore
from naive_bayes import NaiveBayesClassifier
from split import n_docs_per_size, sizes
from utils import choices


b = 1000


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


def learn(training_sets_path: str) -> list:
    """Learn the sixty classifiers.

    Parameters
    ----------
    training_sets_path : str
        Location of the trainingSets directory.

    Returns
    -------
    list[NaiveBayesClassifier]
        A list of learned classifiers.
    """
    systems = []
    for size in sizes:
        for i in range(1, n_docs_per_size + 1):
            path = Path(training_sets_path) / str(size) / f"train{i}.txt"
            x_train, y_train = get_x_y(path)

            clf_count = NaiveBayesClassifier()
            clf_count.fit(x_train, y_train)
            systems.append(clf_count)

            clf_binary = NaiveBayesClassifier(binary=True)
            clf_binary.fit(x_train, y_train)
            systems.append(clf_binary)

    return systems


def bootstrap(test_file: str, systems: list) -> None:
    """Perform bootstraping on the test set and create the results file.

    Parameters
    ----------
    test_file : str
        Location of a test file.
    systems : list[NaiveBayesClassifier]
        List of systems to perform bootstraping on.
    """
    with open(f"./output{b}.csv", "w") as f:
        f.write("pval,effect_size,typeA,typeB\n")
    x_test, y_test = get_x_y(test_file)
    track = 0
    pairs = set()
    for i, clf_a in enumerate(systems):
        for j, clf_b in enumerate(systems):
            if clf_a is clf_b or (i, j) in pairs or (j, i) in pairs:
                continue

            pairs.add((i, j))
            track += 1
            if track % 10 == 0:
                print(f"{track} / {1770} = {round(100 * track / 1770, 3)}%")

            preds_a = clf_a.predict(x_test)
            preds_b = clf_b.predict(x_test)

            f_a = fscore(preds_a, y_test)
            f_b = fscore(preds_b, y_test)
            delta_f = abs(f_a - f_b)

            n = len(preds_a)
            s = 0
            for _ in range(b):
                idx = choices(list(range(n)), [1 for _ in range(n)], n)
                preds_a_ = [preds_a[i] for i in idx]
                preds_b_ = [preds_b[i] for i in idx]
                y_test_ = [y_test[i] for i in idx]

                f_a_ = fscore(preds_a_, y_test_)
                f_b_ = fscore(preds_b_, y_test_)
                delta_f_ = f_a_ - f_b_

                s = s + 1 if delta_f_ >= 2 * abs(delta_f) else s

            pval = s / b

            with open(f"./output{b}.csv", "a") as f:
                line = ",".join(
                    [
                        str(pval),
                        str(delta_f),
                        "b" if clf_a.binary else "c",
                        "b" if clf_b.binary else "c",
                    ]
                )
                f.write(line + "\n")


def main(training_sets_path: str, test_file: str) -> None:
    """Produce the deliverables.

    Parameters
    ----------
    training_sets_path : str
        Path to the training sets directory.
    test_file : str
        Path to the test file.
    """
    systems = learn(training_sets_path)
    bootstrap(test_file, systems)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--training-sets-path",
        action="store",
        default="./trainingSets",
        help="Unix-style path to the trainingSets directory.",
    )
    parser.add_argument(
        "--test-file",
        action="store",
        default="./testMaster.txt",
        help="Unix-style path to the test file.",
    )

    args = parser.parse_args()

    main(args.training_sets_path, args.test_file)
