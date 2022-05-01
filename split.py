"""Create collections randomly selected subsets of the training set.
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil


from utils import choices


# The number of examples in each subset
sizes = (2600, 1300, 650)
# The number of subsets to create for each size
n_docs_per_size = 10


def create_training_splits(train_file: str) -> None:
    """Create several subsets of the training set.

    Parameters
    ----------
    train_file : str
        Unix-style path to the trainMaster.txt file.
    """
    training_sets_path = Path("./trainingSets")
    shutil.rmtree(training_sets_path, ignore_errors=True)
    training_sets_path.mkdir(parents=True)

    with open(train_file, "r") as f:
        lines = f.readlines()

    for size in sizes:
        size_path = training_sets_path / str(size)
        size_path.mkdir()

        for i in range(n_docs_per_size):
            idx = choices(list(range(len(lines))), [1 for i in range(len(lines))], k=size)
            selected = [lines[i] for i in idx]

            with open(size_path / f"train{i+1}.txt", "w") as f:
                f.writelines(selected)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--train-file",
        action="store",
        default="train.txt",
        help="Unix-style path to the trainMaster.txt file.",
    )

    args = parser.parse_args()

    create_training_splits(args.train_file)
