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


class LambdaMeans(Model):

    def __init__(self, cluster_lambda, clustering_training_iterations):
        super().__init__()
        self.cluster_lambda = cluster_lambda
        self.clustering_training_iterations = clustering_training_iterations

    def fit(self, X, _):
        self.num_examples, self.num_input_features = X.shape
        # Initialize mu_1
        self.mu = [np.mean(X.toarray(), axis=0)]
        if not(self.cluster_lambda > 0):
            # Set lambda to default value
            self.cluster_lambda = np.mean(np.linalg.norm(X.toarray() - self.mu[0], ord=2, axis=1))

        for i in range(self.clustering_training_iterations):
            # E step
            r = np.zeros((self.num_examples, len(self.mu)))

            for n in range(self.num_examples):
                xn = X[n, :].toarray()[0]
                norms = np.linalg.norm(self.mu - xn, ord=2, axis=1)

                if (np.min(norms) > self.cluster_lambda):
                    r = np.hstack((r, np.zeros((self.num_examples, 1))))
                    self.mu.append(xn)
                    r[n, len(self.mu)-1] = 1
                else:
                    r[n, np.argmin(norms)] = 1

            # M step
            for k in range(len(self.mu)):
                in_cluster = np.where(r[:, k] == 1)
                if (len(in_cluster) == 0):
                    self.mu[k] = np.zeros(self.num_input_features)
                else:
                    self.mu[k] = np.sum(X.toarray()[in_cluster], axis=0)/len(in_cluster)

    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        predictions = np.zeros(num_examples)

        for n in range(num_examples):
            xn = X[n, :].toarray()[0]
            norms = np.linalg.norm(self.mu - xn, ord=2, axis=1)

            predictions[n] = np.argmin(norms)

        return predictions


class StochasticKMeans(Model):

    def __init__(self, clustering_training_iterations):
        super().__init__()
        self.clustering_training_iterations = clustering_training_iterations

    def fit(self, X, _):
        self.num_examples, self.num_input_features = X.shape


    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        predictions = np.zeros(num_examples)

        return predictions

