import numpy as np
from scipy.special import expit


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class LogisticRegression(Model):

    def __init__(self, online_learning_rate, num_features_to_select, gd_iterations):
        super().__init__()
        self.online_learning_rate = online_learning_rate
        self.num_features_to_select = num_features_to_select
        self.gd_iterations = gd_iterations

    def fit(self, X, y):
        self.num_examples, self.num_input_features = X.shape

        # If we do not select to use all features...
        if (self.num_features_to_select != -1):
            if (self.num_features_to_select > self.num_input_features):
                self.num_features_to_select = self.num_input_features

            # Initialize vector of information gain (technically -h(y|x))
            info_gain = np.zeros(self.num_input_features)

            for j in range(self.num_input_features):

                x_j = X[:, j].toarray().T[0]

                mean = np.mean(x_j)
                # Adjust each feature to 0 or 1 using thresholding
                x_j = np.where(x_j >= mean, 1, 0)

                for yi in [0, 1]:
                    for xj in [0, 1]:
                        # p(x_j)
                        q = len(x_j[x_j == xj])/len(x_j)

                        # p(y_i, x_j)
                        y_xj = y[x_j == xj]
                        p = len(y_xj[y_xj == yi])/len(y)

                        if ((p > 0) & (q > 0)):
                            info_gain[j] += p * np.log(p/q)

            # Gets indices of largest IGs for feature selection
            ind = np.argpartition(info_gain, -self.num_features_to_select)[-self.num_features_to_select:]
        else:
            # Otherwise use all features
            ind = np.arange(self.num_input_features)

        X = X[:, ind]

        # Initialize w to a vector of 0s to be trained using only top features
        w = np.zeros(len(ind))

        for i in range(self.gd_iterations):

            # Matrices in summation for gradient calculation
            summand_1 = np.multiply(np.multiply(expit(np.dot(X.toarray(), -w)), y), X.toarray().T).T
            summand_2 = np.multiply(np.multiply(expit(np.dot(X.toarray(), w)), 1-y), -X.toarray().T).T

            # Calculates summation to get vector form of gradient of w
            w_grad = np.sum(summand_1 + summand_2, axis = 0)

            w += self.online_learning_rate * w_grad

        # insert 0s in w for unused features
        w_2 = np.zeros(self.num_input_features)
        w_2[ind] = w

        self.w = w_2

    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        predictions = expit(np.dot(X.toarray(), self.w))

        y_hat = np.where(predictions >= 0.5, 1, 0)

        return y_hat

    # Returns the vector of information gain
    def ig(self, X, y):
        num_examples, num_input_features = X.shape
        # Initialize vector of information gain
        ig = np.zeros(num_input_features)

        for j in range(num_input_features):

            x_j = X[:, j].toarray().T[0]

            mean = np.mean(x_j)
            np.where(x_j >= mean, 1, 0)

            for yi in [0, 1]:
                for xj in [0, 1]:
                    # p(x_j)
                    q = len(x_j[x_j == xj])/len(x_j)

                    # p(y_i, x_j)
                    y_xj = y[x_j == xj]
                    p = len(y_xj[y_xj == yi])/len(y)

                    if ((p > 0) & (q > 0)):
                        ig[j] -= p * np.log(p/q)

        return ig

# TODO: Add other Models as necessary.
