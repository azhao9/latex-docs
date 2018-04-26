import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y, **kwargs):
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

    def fit(self, X, y, **kwargs):
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

    def __init__(self):
        super().__init__()

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']

        self.num_examples, self.num_input_features = X.shape
        X = X.toarray()
        self.mu = [np.mean(X, axis=0)]

        if not(lambda0 > 0):
            lambda0 = np.mean(np.linalg.norm(X - self.mu[0], axis=1), axis=0)

        for i in range(iterations):
            r = np.zeros((self.num_examples, len(self.mu)))

            # E step
            for n in range(self.num_examples):
                xn = X[n, :]
                norms = np.linalg.norm(self.mu - xn, axis=1)

                if (np.min(norms) > lambda0):
                    self.mu.append(xn)
                    r = np.hstack((r, np.zeros((self.num_examples, 1))))
                    r[n, len(self.mu)-1] = 1
                else:
                    r[n, np.argmin(norms)] = 1

            # M step
            for k in range(len(self.mu)):
                in_cluster = np.where(r[:, k] == 1)
                if (len(in_cluster) == 0):
                    self.mu[k] = np.zeros(self.num_input_features)
                else:
                    self.mu[k] = np.mean(X[in_cluster], axis=0)

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

        X = X.toarray()
        predictions = np.zeros(num_examples)

        for n in range(num_examples):
            xn = X[n, :]
            norms = np.linalg.norm(self.mu - xn, axis=1)

            predictions[n] = np.argmin(norms)
        
        return predictions


class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']

        self.num_examples, self.num_input_features = X.shape
        X = X.toarray()
        self.centers = [None] * num_clusters

        if (num_clusters == 1):
            self.centers = np.mean(X, axis=0)
        else:
            min_center = X.min(0)
            max_center = X.max(0)

            for k in range(num_clusters):
                self.centers[k] = (k * max_center + (num_clusters - k) * min_center) / num_clusters

        for i in range(iterations):
            p = np.zeros((self.num_examples, num_clusters))

            c = 2
            beta = c * (i+1)

            # E step
            for n in range(self.num_examples):
                xn = X[n, :]
                norms = np.linalg.norm(self.centers - xn, axis=1)

                d_hat = np.mean(norms, axis=0)

                p[n, :] = np.exp(-beta * norms / d_hat)/np.sum(np.exp(-beta * norms / d_hat)) 
            
            # M step
            for k in range(num_clusters):
                pk = p[:, k]

                self.centers[k] = np.dot(X.T, p[:, k])/np.sum(p[:, k])

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

        X = X.toarray()
        predictions = np.zeros(num_examples)

        for n in range(num_examples):
            xn = X[n, :]
            norms = np.linalg.norm(self.centers - xn, axis=1)

            predictions[n] = np.argmin(norms)
        
        return predictions
