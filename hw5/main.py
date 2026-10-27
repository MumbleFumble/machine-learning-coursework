"""
This file implements a program to read data in, build a model
and print results to stdout.

Besides implementing explore_dataset, you do not need to make edits to this
file. You can add debugging code or code to help produce your report, but 
this code should not be run by default in your final submission.

Brown cs1420, Spring 2026
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from get_data import get_data
from matplotlib.axes import Axes
from models import DecisionTree, node_score_entropy, node_score_error, node_score_gini


def print_tree_loss(
    tree: DecisionTree,
    pruned_tree: DecisionTree,
    train_data: np.ndarray,
    test_data: np.ndarray,
):
    """
    Code to print out the different losses using a tree and pruned tree.
    You do not need to change this code!
    """
    print("No-pruning Training data loss:", tree.loss(train_data))
    print("No-pruning Test data loss:", tree.loss(test_data))
    print("Pruning Training data loss:", pruned_tree.loss(train_data))
    print("Pruning Testing loss:", pruned_tree.loss(test_data))
    if (
        tree.loss(train_data)
        <= pruned_tree.loss(train_data)
        <= pruned_tree.loss(test_data)
        <= tree.loss(test_data)
    ):
        print("Building a pruned tree successfully!")


def loss_plot(
    ax: Axes,
    title: str,
    tree: DecisionTree,
    pruned_tree: DecisionTree,
    train_data: np.ndarray,
    test_data: np.ndarray,
) -> None:
    """
    Example plotting code. This plots four curves: the training and testing
    average loss using tree and pruned tree.
    You do not need to change this code!

    Parameters
    ----------
    ax: A matplotlib Axes instance.
    title: str
        A title for the graph.
    tree: DecisionTree
        An unpruned DecisionTree instance.
    pruned_tree: DecisionTree
        A pruned DecisionTree instance.
    train_data: np.ndarray
        Training dataset returned from get_data.
    test_data: np.ndarray
        Test dataset returned from get_data.

    Returns
    -------
    None
    """
    fontsize = 8
    ax.plot(tree.loss_plot_vec(train_data), label="train non-pruned")
    ax.plot(tree.loss_plot_vec(test_data), label="test non-pruned")
    ax.plot(pruned_tree.loss_plot_vec(train_data), label="train pruned")
    ax.plot(pruned_tree.loss_plot_vec(test_data), label="test pruned")

    ax.locator_params(nbins=3)
    ax.set_xlabel("number of nodes", fontsize=fontsize)
    ax.set_ylabel("loss", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc="upper center", shadow=True, fontsize=fontsize - 2)


def explore_dataset(filename: str, class_name: str) -> None:
    """
    Function to print loss values and plot figures.

    Parameters
    ----------
    filename : str
        Name of the file containing data.
    class_name : str
        Column name for which we want to convert to binary entries.

    Returns
    -------
    None
    """
    train_data, validation_data, test_data = get_data(filename, class_name)
    # print(train_data)
    # print(validation_data)
    # print(test_data)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (node score error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

    for gain_fn in [node_score_error, node_score_entropy, node_score_gini]:
        tree = DecisionTree(train_data, gain_function=gain_fn)
        pruned = DecisionTree(train_data, validation_data, gain_function=gain_fn)
        print_tree_loss(tree, pruned, train_data, test_data)
    
    # REPORT QUESTION (b) - uncomment to generate plot, comment out before submitting
    if "spam" in filename:
        losses = []
        for depth in range(1, 16):
            tree = DecisionTree(train_data, gain_function=node_score_entropy, max_depth=depth)
            losses.append(tree.loss(train_data))
        plt.figure()
        plt.plot(range(1, 16), losses)
        plt.xlabel("Max Depth")
        plt.ylabel("Training Loss")
        plt.title("Spam: Training Loss vs Max Depth (Entropy)")
        plt.savefig("depth_vs_loss.png")
        plt.show()


def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset("data/chess.csv", "won")
    explore_dataset("data/spam.csv", "1")


main()
