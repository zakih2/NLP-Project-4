"""Partition the training data into train and test sets.
"""

from argparse import ArgumentParser
from pathlib import Path
import random


# The number of examples to allocate to the test set
test_size = 400


def create_train_test_split(data_file: str, train_file: str, test_file: str) -> None:
    """Create the train and test sets.

    Parameters
    ----------
    data_file : str
        Unix-style path to the fulldataLabeled.txt file.
    train_file : str
        Unix-style path to the trainMaster.txt file.
    test_file : str
        Unix-style path to the testMaster.txt file.
    """
    train_file: Path = Path(train_file)
    test_file: Path = Path(test_file)

    train_file.unlink(missing_ok=True)
    test_file.unlink(missing_ok=True)

    train_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "r") as f:
        lines = f.readlines()

    test_idx = set(random.sample(range(len(lines)), test_size))
    train_idx = set(range(len(lines))).difference(test_idx)

    test_lines = [l for i, l in enumerate(lines) if i in test_idx]
    train_lines = [l for i, l in enumerate(lines) if i in train_idx]

    with open(train_file, "w") as f:
        f.writelines(train_lines)
    with open(test_file, "w") as f:
        f.writelines(test_lines)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--data-file",
        action="store",
        default="fulldataLabeled.txt",
        help="Unix-style path to the fulldataLabeled.txt file.",
    )
    parser.add_argument(
        "--train-file",
        action="store",
        default="train.txt",
        help="Unix-style path to the trainMaster.txt file.",
    )
    parser.add_argument(
        "--test-file",
        action="store",
        default="test.txt",
        help="Unix-style path to the testMaster.txt file.",
    )

    args = parser.parse_args()

    create_train_test_split(args.data_file, args.train_file, args.test_file)
