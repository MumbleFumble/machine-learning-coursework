from __future__ import annotations

from typing import Callable

import numpy as np


def l2_loss(predictions: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes L2 loss (sum squared loss) between predictions and Y.

    Parameters
    ----------
    predictions : np.ndarray
        A 1D Numpy array.
    Y : np.ndarray
        A 1D Numpy array of the same size as predictions.

    Returns
    -------
    float
        L2 loss using between predictions and Y.
    """
    # TODO: Add your solution code here.
    return np.sum((predictions - Y) ** 2)


def sigmoid(a: np.ndarray) -> np.ndarray:
    """
    Sigmoid function, given by sigma(a) =  1/(1 + exp(-a)), applied component-wise

    Parameters
    ----------
    a : np.ndarray
        A Numpy array.

    Returns
    -------
    np.ndarray
        Sigmoid function evaluation on a.
    """
    return np.where(a > 0, 1 / (1 + np.exp(-a)), np.exp(a) / (np.exp(a) + np.exp(0)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """
    First derivative of the sigmoid function with respect to a.

    Parameters
    ----------
    a : np.ndarray
        A Numpy array.

    Returns
    -------
    np.ndarray
        Sigmoid derivative evaluation on a.
    """
    # TODO: calculate the first derivative of the sigmoid function
    s = sigmoid(a)
    return s * (1 - s)


class OneLayerNN:
    """
    Constructor for a one layer neural network.

    Attributes
    ----------
    w : np.ndarray
        The weights of the first layer of the neural network model.
    learning_rate : float
        The learning rate to use for SGD.
    epochs : int
        The number of times to pass through the dataset.
    o : np.ndarray
        The output of the network.
    """

    def __init__(self):
        """
        Initializes a one layer neural network.
        """
        # initialize self.w in train()
        self.w = None
        self.learning_rate = 0.001
        self.epochs = 25
        self.o = None

    def train(self, X: np.ndarray, Y: np.ndarray, print_loss: bool = True) -> None:
        """
        Training loop with SGD for OneLayerNN model.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.
        print_loss : bool, default=True
            If True, print the loss after each epoch.

        Returns
        -------
        None
        """
        # TODO: initialize self.w. In the OneLayerNN, you can assume that every example
        # has a 1 as the last feature, so no separate bias term is needed.
        # TODO: Train network for certain number of epochs defined in self.epochs
        # TODO: Use a variable called epoch to denote the current epoch's number
            # TODO: Shuffle the examples (X) and labels (Y)
            # TODO: We need to iterate over each example for each epoch
            # (This is equivalent to using a batch size of one.)
                # TODO: Perform the forward and backward pass on the current example
            # # Print the loss after every epoch
            # if print_loss:
            #     print("Epoch: {} | Loss: {}".format(epoch, self.loss(X, Y)))
        input_size = X.shape[1]
        self.w = np.random.uniform(-0.1, 0.1, input_size)
        for epoch in range(self.epochs): # epochs is 1, from description
            # Shuffle the examples and labels together
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
 
            for i in range(X_shuffled.shape[0]):
                self.forward_pass(X_shuffled[i])
                self.backward_pass(X_shuffled[i], Y_shuffled[i])
 
            if print_loss:
                print("Epoch: {} | Loss: {}".format(epoch, self.loss(X, Y)))

    def forward_pass(self, x: np.ndarray) -> None:
        """
        Computes the output of the network given an input vector x and stores the result in self.o.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.

        Returns
        -------
        None
        """
        # TODO: Calculate output of neural network on x
        self.o = np.dot(self.w, x)

    def backward_pass(self, x: np.ndarray, y: float) -> None:
        """
        First computes the gradient of the loss on an example with respect to self.w.
        Then updates self.w. Should only be called after self.forward_pass.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.
        y : float
            Scalar, the label for the example x.

        Returns
        -------
        None
        """
        # TODO: Calculate the gradient of the weights
        # # TODO: Update the weights using the gradient
        # dL/dw = 2*(o - y) * x
        grad = 2 * (self.o - y) * x
        self.w -= self.learning_rate * grad

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the total squared error on some dataset (X, Y).

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the squared error of the model on the dataset.
        """
        # Perform the forward pass and compute the l2 loss
        outputs = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            outputs[i] = self.o

        return l2_loss(outputs, Y)

    def average_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the mean squared error on some dataset (X, Y).
        MSE = Total squared error / # of examples

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the mean squared error of the model on the dataset.
        """
        return self.loss(X, Y) / X.shape[0]


class TwoLayerNN:
    """
    Two layer neural network trained with Stochastic Gradient Descent (SGD).

    Attributes
    ----------
    activation : Callable
        The activation function applied after the first layer.
    activation_derivative : Callable
        The derivative of the activation function. Used for training.
    hidden_size : int
        The hidden size of the network.
    learning_rate : float
        The learning rate to use for SGD.
    epochs : int
        The number of times to pass through the dataset.

    Other variable naming conventions:
        Letters:
            w = weights layers
            b = bias layers
            a = output of the first layer computed during forward pass
            o = the activated output of the first layer computed during forward pass
                (ie, result of applying activation function to an a)

    Numbers serve as layer indices. For example, w12 would be the weights between
    the first and second layer.
    """

    def __init__(
        self,
        hidden_size: int,
        activation: Callable = sigmoid,
        activation_derivative: Callable = sigmoid_derivative,
    ):
        """
        Constructor for a two layer neural network.

        Parameters
        ----------
        hidden_size : int
            The hidden size of the network.
        activation : Callable, default=sigmoid
            The activation function applied after the first layer.
        activation_derivative : Callable, default=sigmoid_derivative
            The derivative of the activation function. Used for training.
        """
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size
        self.learning_rate = 0.01
        self.epochs = 30

        # initialize the following weights and biases in the train() method
        self.W01 = None
        self.b1 = None
        self.W12 = None
        self.b2 = None

        # Outputs of each layer
        self.a1 = None
        self.o1 = None
        self.a2 = None
        self.o2 = None

    def train(self, X: np.ndarray, Y: np.ndarray, print_loss: bool = True) -> None:
        """
        Trains the TwoLayerNN with SGD using Backpropagation.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.
        print_loss : bool, default=True
            If True, print the loss after each epoch.

        Returns
        -------
        None
        """
        # NOTE:
        # Use numpy arrays of the following dimensions for your model's parameters.
        # layer 1 weights (W01): hidden_size x input_size (2D matrix)
        # layer 1 bias (b1): hidden_size (1D vector)
        # layer 2 weights (W12): 1 x hidden_size (2D matrix)
        # layer 2 bias (b2): 1 (1D vector)
        # HINT: for best performance initialize weights with np.random.normal or np.random.uniform

        # TODO: Weight and bias initialization
        # TODO: Train network for certain number of epochs
        # TODO: Use a variable called epoch to denote the current epoch's number
            # TODO: Shuffle the examples (X) and labels (Y)
            # TODO: We need to iterate over each example for each epoch
            # (This is equivalent to using a batch size of one.)
                # TODO: Perform the forward and backward pass on the current example
            # # Print the loss after every epoch
            # if print_loss:
            #     print("Epoch: {} | Loss: {}".format(epoch, self.loss(X, Y)))
        input_size = X.shape[1]
 
        # Initialize 
        self.W01 = np.random.uniform(-0.1, 0.1, (self.hidden_size, input_size))
        self.b1  = np.zeros(self.hidden_size)
        self.W12 = np.random.uniform(-0.1, 0.1, (1, self.hidden_size))
        self.b2  = np.zeros(1)

        for epoch in range(self.epochs): #same as above
            # Shuffle examples and labels together
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
 
            for i in range(X_shuffled.shape[0]):
                self.forward_pass(X_shuffled[i])
                self.backward_pass(X_shuffled[i], Y_shuffled[i])
 
            if print_loss:
                print("Epoch: {} | Loss: {}".format(epoch, self.loss(X, Y)))

    def forward_pass(self, x: np.ndarray) -> None:
        """
        Computes the outputs of the network given an input vector x. Stores the activation function input
        (weighted sum for each neuron) for each layer in self.a* and the post-activation output for each
        layer in self.o*.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.

        Returns
        -------
        None
        """
        # TODO: Calculate output of neural network on X
        self.a1 = self.W01 @ x + self.b1
        self.o1 = self.activation(self.a1)
 
        # 2 
        self.a2 = self.W12 @ self.o1 + self.b2
        self.o2 = self.a2[0] 


    def backprop(
        self, x: np.ndarray, y: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the gradients of the loss w.r.t the weights and biases of each layer
        using the backpropagation algorithm.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.
        y : float
            Scalar, the label for the example x.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (W12_grad, b2_grad, W01_grad, b1_grad)
        """
        # TODO: Implement backpropagation
        # Hint: Use the values computed and saved in self.forward_pass

        # dL/do2 = 2*(o2 - y)
        # dL/dW12[1,i] = dL/do2 * o1[i]
        # dL/db2 = dL/do2
        # dL/do1[i] = dL/do2 * W12[1,i]
        # dL/da1[i] = dL/do1[i] * sigma'(a1[i])
        # dL/dW01[i,j] = dL/da1[i] * x[j]
        # dL/db1[i] = dL/da1[i]
        dL_do2   = 2 * (self.o2 - y)
        W12_grad = dL_do2 * self.o1.reshape(1, -1)  
        b2_grad  = np.array([dL_do2])
        dL_do1   = dL_do2 * self.W12[0] 
        dL_da1   = dL_do1 * self.activation_derivative(self.a1)
        W01_grad = np.outer(dL_da1, x)
        b1_grad  = dL_da1  
        return W12_grad, b2_grad, W01_grad, b1_grad

    def backward_pass(self, x: np.ndarray, y: float) -> None:
        """
        First computes the gradient of the loss on an example with respect to self.W01, self.b1, self.W12,
        and self.b2 by calling self.backprop. Then updates all those parameters. Should only be called
        after self.forward_pass.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.
        y : float
            Scalar, the label for the example x.

        Returns
        -------
        None
        """
        # TODO: Compute the gradients for the model's weights by calling self.backprop
        # TODO: Update the weights using gradient descent
        W12_grad, b2_grad, W01_grad, b1_grad = self.backprop(x, y)
        self.W12 -= self.learning_rate * W12_grad
        self.b2  -= self.learning_rate * b2_grad
        self.W01 -= self.learning_rate * W01_grad
        self.b1  -= self.learning_rate * b1_grad

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the total squared error on some dataset (X, Y).

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the squared error of the model on the dataset.
        """
        # Perform the forward pass and compute the l2 loss
        outputs = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            outputs[i] = self.o2

        return l2_loss(outputs, Y)

    def average_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the mean squared error on some dataset (X, Y).
        MSE = (Total squared error) / (# of examples)

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the mean squared error of the model on the dataset.
        """
        return self.loss(X, Y) / X.shape[0]
