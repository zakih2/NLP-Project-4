"""
Create plots of pvalues and effect sizes output by the classification system.

Notes
-----
This software expects the following columns in the input_file .csv file:
    pval - the pvalue
    effect_size - the effect size
    typeA - a symbol indicating the type of the first system, e.g., "c"
    typeB - a symbol indicating the type of the second system, e.g., "b"

This slightly contradicts that specifications under D3, but it is convenient.
"""

from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


def pval_vs_effect_basic(input_file: str, output_file: str) -> None:
    """Plot 1-pvalues vs effect size.

    Parameters
    ----------
    input_file : str
        Location of an input file of data.
    output_file : str
        Output file location for the plot.
    """
    df = pd.read_csv(input_file)

    pvals = 1 - df["pval"].to_numpy()
    effect_sizes = df["effect_size"].to_numpy()

    plt.scatter(effect_sizes, pvals, marker="x", color="black")
    plt.title("1-pvalue vs effect size")
    plt.ylabel("1-pvalue")
    plt.xlabel("Effect size")
    plt.savefig(output_file, dpi=400)

    plt.clf()


def pval_vs_effect_system(input_file: str, output_file: str) -> None:
    """Plot 1-pvalue vs effect size, with different markers.

    Parameters
    ----------
    input_file : str
        Location of an input file of data.
    output_file : str
        Output file location for the plot.
    """
    df = pd.read_csv(input_file)

    pvals = 1 - df["pval"].to_numpy()
    effect_sizes = df["effect_size"].to_numpy()
    same_system = [a == b for a, b in zip(df["typeA"], df["typeB"])]

    # Plot the points which correspond to the same systems
    _pvals = [p for i, p in enumerate(pvals) if same_system[i]]
    _effect_sizes = [p for i, p in enumerate(effect_sizes) if same_system[i]]
    plt.scatter(_effect_sizes, _pvals, marker="x", color="blue", label="Same systems")
    # Plot the points which correspond to the different systems
    _pvals = [p for i, p in enumerate(pvals) if not same_system[i]]
    _effect_sizes = [p for i, p in enumerate(effect_sizes) if not same_system[i]]
    plt.scatter(_effect_sizes, _pvals, marker="o", color="red", label="Diff systems")

    plt.title("1-pvalue vs effect size")
    plt.ylabel("1-pvalue")
    plt.xlabel("Effect size")
    plt.legend()
    plt.savefig(output_file, dpi=400)

    plt.clf


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--input-file",
        action="store",
        default="./output.csv",
        help="Unix-style path to a .csv file containing data.",
    )
    parser.add_argument(
        "--plot1-file",
        action="store",
        default="./plot1.png",
        help="Unix-style path to the output file for plot1.",
    )
    parser.add_argument(
        "--plot2-file",
        action="store",
        default="./plot2.png",
        help="Unix-style path to the output file for plot2.",
    )

    args = parser.parse_args()

    pval_vs_effect_basic(args.input_file, args.plot1_file)
    pval_vs_effect_system(args.input_file, args.plot2_file)
