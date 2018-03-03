import numpy as np


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



            # Initialize w to a vector of 0s to be trained
        self.w = np.zeros(self.num_input_features)

    def predict(self, X):
        # TODO: Write code to make predictions.
        pass

    def g(z):
        return 1 / (1 + np.exp(-z))

    # Returns the vector of information gain
    def ig(X, y):
        num_examples, num_input_features = X.shape
        # Initialize vector of information gain
        ig = np.zeros(num_input_features)

        for j in range(num_input_features):
            # x_j is the array of feature values for feature j
            x_j = X[:, j].toarray()[0]

            if (len(np.unique(x_j)) > 2):
                # More than 2 distinct values for feature j, so we take it to be continuous
                # Modify x_j to make everything 0 and 1

                mean = np.mean(x_j)
                # Initialize x_j_binary to vector of 0s
                x_j_binary = np.zeros(self.num_examples)
                # Change everywhere x_j is at least the mean to 1 in x_j_binary
                x_j_binary[np.where(x_j >= mean)] = 1
                # Sets x_j to the binary version
                x_j = x_j_binary

            # possible values x_j and y can take on
            xvals = [0, 1]
            yvals = [0, 1]

            for xj in xvals:
                for yi in yvals:
                    # Conditional probability
                    y_xj = y[np.where(x_j == xj)]
                    p = len(y_xj[y_xj == yi])/len(y_xj)

                    # only update ig if p is non-zero since otherwise log is undefined
                    if (p > 0):
                        ig[j] -= p * np.log(p)

        return ig

# TODO: Add other Models as necessary.
