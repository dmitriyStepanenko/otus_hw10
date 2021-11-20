import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            learning_rate: float = 1e-3,
            reg: float = 1e-5,
            num_iters: int = 100,
            batch_size: int = 200,
            verbose: bool = False
    ):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in np.arange(num_iters):

            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update

            self.w -= gradW * learning_rate

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self.loss_history

    def predict_proba(self, X, append_bias=False) -> np.ndarray:
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)

        y_proba = np.vstack((1 - self.sigma(X), self.sigma(X)))

        return y_proba

    def predict(self, X):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = y_proba.argmax(axis=0)

        return y_pred

    def loss(
            self,
            X_batch: csr_matrix,
            y_batch: np.ndarray,
            reg: float
    ):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        num_train = X_batch.shape[0]
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0
        # Compute loss and gradient. Your code should not contain python loops.

        sig = self.sigma(X_batch)
        loss = - np.dot(y_batch, np.log(sig)) - np.dot((1 - y_batch), np.log(1 - sig))

        dw = X_batch.T.dot(sig-y_batch)

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.
        loss /= num_train
        dw /= num_train

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.

        loss += (reg / (2 * num_train)) * np.sum(np.square(self.w))
        dw[1:] += (reg / num_train) * self.w[1:]

        return loss, dw

    @staticmethod
    def append_biases(X) -> csr_matrix:
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()

    def sigma(self, x: csr_matrix):
        return np.power(1 + np.exp(-x.dot(self.w)), -1)
