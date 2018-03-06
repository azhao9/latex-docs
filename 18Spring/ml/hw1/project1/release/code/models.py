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


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        num_examples, num_input_features = X.shape

        # Creates a vector of -1, 0, 1 for dot product.
        # If odd number of features, omit middle feature. 
        half_length = (int)(num_input_features/2)
        if(num_input_features % 2 == 1):
            w = np.concatenate((np.ones(half_length), [0], -1 * np.ones(half_length)))
        else:
            w = np.concatenate((np.ones(half_length), -1 * np.ones(half_length)))

        # If dot product result is >=0, sum of first half is greater
        # Otherwise, sum of first half is less.
        prod = X.dot(w)
        
        # Convert to an array of 1s and 0s
        y_hat = (prod >= 0).astype(int)
        return y_hat

class Perceptron(Model):

    def __init__(self, learning_rate, training_iterations):
        super().__init__()
        self.learning_rate = learning_rate
        self.training_iterations = training_iterations

    def fit(self, X, y):
        self.num_examples, self.num_input_features = X.shape

        # Initialize w to vector of 0s to be trained
        self.w = np.zeros(self.num_input_features)

        # Initialize y_hat to vector of 0s
        y_hat = np.zeros(self.num_examples)

        # Modify y to have 0s replaced with -1s
        y[y == 0] = -1

        for k in range(self.training_iterations):
            for i in range(self.num_examples):
                # Gets feature i as an array
                x_i = X[i, :].toarray()[0]

                prod = np.dot(x_i, self.w)

                if (prod >= 0): 
                    y_hat[i] = 1
                else:
                    y_hat[i] = -1

                # Adjusts w vector
                if (y_hat[i] != y[i]):
                    self.w += self.learning_rate * y[i] * x_i

    def predict(self, X):
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

        y_hat = np.zeros(num_examples)

        for i in range(num_examples):
            # Gets feature i as an array
            x_i = X[i, :].toarray()[0]

            prod = np.dot(x_i, self.w)

            # Returns sign of the dot product
            if (prod >= 0):
                y_hat[i] = 1
            else:
                y_hat[i] = 0

        return y_hat

# TODO: Add other Models as necessary.
