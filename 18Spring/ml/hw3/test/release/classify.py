import os
import argparse
import sys
import pickle
import numpy as np

from data import load_data
from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

'''
def load_data(filename):
    instances = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
            
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances
'''

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    parser.add_argument("--num-boosting-iterations", type=int, help="The number of boosting iterations to run.", default=10)
    
    
    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")

class Adaboost(Predictor):
    def __init__(self, num_boosting_iterations):
        self.num_boosting_iterations = num_boosting_iterations
        self.j = np.zeros(num_boosting_iterations)
        self.c = np.zeros(num_boosting_iterations)
        self.alpha = np.zeros(num_boosting_iterations)

    def train(self, X, y):
        self.num_examples, self.num_input_features = X.shape

        # Initialize uniform distribution
        D = np.ones(self.num_examples)/self.num_examples

        # Set default stop hypothesis to use all iterations
        stop_hypothesis = self.num_boosting_iterations

        for t in range(self.num_boosting_iterations):
            # Initialize error along each feature
            error_j = [None] * self.num_examples

            for j in range(self.num_input_features):
                x = X[:, j].toarray()
                unique = np.sort(np.unique(x))

                error_c = np.zeros(len(unique))

                for k in range(len(unique)):
                    for i in range(self.num_examples):
                        if (X[i, j] > unique[k]):
                            correct = np.where(x > unique[k], y, 0)
                        else:
                            correct = np.where(x <= unique[k], y, 0)

                        if (np.sum(correct) >= 0):
                            y_hat = 1
                        else:
                            y_hat = -1

                        if (y_hat != y[i]):
                            error_c[k] += D[i]

                error_j[j] = error_c

            # Finding index for argmin j
            min_inds = [np.argmin(error_j[j]) for j in range(self.num_features)]
            min_feature_ind = np.argmin([error_j[j][min_inds[j]] for j in range(self.num_features)])
            self.j[t] = min_feature_ind

            # Finding value for argmin c
            min_c_ind = np.argmin(error_j[min_feature_ind])
            self.c[t] = np.sort(np.unique(X[:, min_feature_ind].toarray()))[min_c_ind]

            # error_t given by the smallest error
            error_t = error_j[min_feature_ind][min_c_ind]

            if (error_t < 0.000001):
                # stop and only use up to previous hypothesis
                stop_hypothesis = t
                continue
        
            self.alpha[t] = 0.5 * np.log((1-error_t)/error_t)

            # Calculate normalizing factor
            h = np.where(X[:, self.j[t]] > c, 1, -1)
            Z = np.dot(D, np.exp(-self.alpha[t] * y * h))
            D *= 1/Z * np.exp(-self.alpha[t] * y * h)

        # Truncate values if we stopped early
        self.j = self.j[:stop_hypothesis]
        self.c = self.c[:stop_hypothesis]
        self.alpha = self.alpha[:stop_hypothesis]

    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        predictions = np.zeros(num_examples)

        for i in range(num_examples):
            votes = np.where(x[self.j] > self.c, self.alpha, -self.alpha)
            if (np.sum(votes) >= 0):
                predictions[i] = 1
            else:
                predictions[i] = 0

        return predictions

def train(X, y, args):
    if (args.algorithm.lower() == 'adaboost'):
        predictor = Adaboost(args.num_boosting_iterations)
    # TODO This is where you will add new algorithms that will subclass Predictor
    
    predictor.train(X, y)

    return predictor


def write_predictions(predictor, X, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
        
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        X, y = load_data(args.data)

        # Train the model.
        predictor = train(X, y, args)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        X, y = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        y_hat = predictor.predict(X)
        invalid_label_mask = (y_hat != 0) & (y_hat != 1) 
        if any(invalid_label_mask):
            raise Exception('All predictions must be 0 or 1, but found other predictions.')
        np.savetxt(args.predictions_file, y_hat, fmt='%d')
        #write_predictions(predictor, X, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

