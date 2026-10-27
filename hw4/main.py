import random
import numpy as np
import pandas as pd
from models import RegularizedLogisticRegression


def extract() -> None:
    """
    Extracts training, validation, & testing data.
    """
    X_train = pd.read_csv("./data/X_train.csv", header=None)
    Y_train = pd.read_csv("./data/y_train.csv", header=None)
    X_val = pd.read_csv("./data/X_val.csv", header=None)
    Y_val = pd.read_csv("./data/y_val.csv", header=None)
    X_test = pd.read_csv("./data/X_test.csv", header=None)
    Y_test = pd.read_csv("./data/y_test.csv", header=None)

    Y_train = np.array([i[0] for i in Y_train.values])
    Y_val = np.array([i[0] for i in Y_val.values])
    Y_test = np.array([i[0] for i in Y_test.values])

    X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_val = np.append(X_val, np.ones((len(X_val), 1)), axis=1)
    X_test = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def main():
    """
    Main driving function: trains model using regularized logistic regression and plots
    results against different lambda values.
    """
    X_train, X_val, X_test, Y_train, Y_val, Y_test = extract()
    X_train_val = np.concatenate((X_train, X_val))
    Y_train_val = np.concatenate((Y_train, Y_val))

    RR = RegularizedLogisticRegression()
    RR.train(X_train, Y_train)
    print(f"Train Accuracy: {RR.accuracy(X_train, Y_train)}")
    print(f"Validation Accuracy: {RR.accuracy(X_val, Y_val)}")

    # TODO: Once runTrainValSplit and runKFold are implemented in models.py, uncomment
    # the following lines of code to plot different lambda values against training/validation errors.

    lambda_list = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    train_errors, val_errors = RR.runTrainValSplit(lambda_list, X_train, Y_train, X_val, Y_val)
    k_fold_errors = RR.runKFold(lambda_list, X_train_val, Y_train_val, 3)
    RR.plotError(lambda_list, train_errors, val_errors, k_fold_errors)

    # TODO: Once you've determined & set the optimal lambda value from the plots, uncomment
    # the following line to see the test accuracy.
    print(f"Test Accuracy: {RR.accuracy(X_test, Y_test)}")

if __name__ == "__main__":
    random.seed(16)
    np.random.seed(16)
    main()
