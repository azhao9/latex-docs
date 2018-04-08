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
        self.j = np.zeros(num_boosting_iterations, dtype=int)
        self.c = np.zeros(num_boosting_iterations)
        self.alpha = np.zeros(num_boosting_iterations)
        self.upper = np.zeros(num_boosting_iterations)
        self.lower = np.zeros(num_boosting_iterations)

    def train(self, X, y):
        self.num_examples, self.num_input_features = X.shape

        # Relabel y with 1 and -1
        y = np.where(y == 0, -1, 1)

        # Initialize uniform distribution
        D = np.ones(self.num_examples)/self.num_examples

        # Set default stop hypothesis to use all iterations
        stop_hypothesis = self.num_boosting_iterations

        for t in range(self.num_boosting_iterations):
            # Initialize values associated with the minimum error
            min_error = float('inf')
            min_c = 0
            min_j = 0
            min_upper = 0
            min_lower = 0
            min_predictions = np.zeros(self.num_examples)

            for j in range(self.num_input_features):
                x = X[:, j].toarray().T[0]

                for c in np.unique(x):
                    # Prediction if x_ij > c
                    if (np.sum(np.where(x > c, y, 0)) >= 0):
                        upper = np.where(x > c, 1, 0)
                        u = 1
                    else:
                        upper = np.where(x > c, -1, 0)
                        u = -1

                    # Prediction if x_ij <= c
                    if (np.sum(np.where(x <= c, y, 0)) >= 0):
                        lower = np.where(x <= c, 1, 0)
                        l = 1
                    else:
                        lower = np.where(x <= c, -1, 0)
                        l = -1

                    predictions = upper + lower

                    error = np.sum(np.where(predictions != y, D, 0))

                    if (error < min_error):
                        min_error = error
                        min_c = c
                        min_j = j
                        min_upper = u
                        min_lower = l
                        min_predictions = predictions

            '''
            # Finds c and j with minimum error
            min_c_ind, min_feature_ind = np.unravel_index(error.argmin(), error.shape)
            self.c[t] = X[min_c_ind, min_feature_ind]
            self.j[t] = min_feature_ind
            '''

            self.c[t] = min_c
            self.j[t] = min_j
            self.upper[t] = min_upper
            self.lower[t] = min_lower
            
            if (min_error < 0.000001):
                # stop and only use up to previous hypothesis
                stop_hypothesis = t
                break
        
            self.alpha[t] = 0.5 * np.log((1-min_error)/min_error)

            '''
            # Determine how h_jc predicts
            x_j = X[:, self.j[t]].toarray().T[0]
            if (np.sum(np.where(x_j > self.c[t], y, 0) >= 0)):
                upper = np.where(x_j > self.c[t], 1, 0)
                #self.upper[t] = 1
            else:
                upper = np.where(x_j > self.c[t], -1, 0)
                #self.upper[t] = -1

            if (np.sum(np.where(x_j <= self.c[t], y, 0) >= 0)):
                lower = np.where(x_j <= self.c[t], 1, 0)
                #self.lower[t] = 1
            else:
                lower = np.where(x_j <= self.c[t], -1, 0)
                #self.lower[t] = -1   

            predictions = upper + lower
            '''

            # Calculate normalizing factor
            Z = np.dot(D, np.exp(-self.alpha[t] * y * min_predictions))
            D *= 1/Z * np.exp(-self.alpha[t] * y * min_predictions)

        # Truncate values if we stopped early
        self.j = self.j[:stop_hypothesis]
        self.c = self.c[:stop_hypothesis]
        self.alpha = self.alpha[:stop_hypothesis]
        self.upper = self.upper[:stop_hypothesis]
        self.lower = self.lower[:stop_hypothesis]

    def predict(self, X):
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        predictions = np.zeros(num_examples)

        for i in range(num_examples):
            x_i = X[i, :].toarray()[0]
            votes = np.zeros(len(self.alpha))

            for t in range(len(self.alpha)):
                if (x_i[self.j[t]] > self.c[t]):
                    votes[t] = self.upper[t]
                else:
                    votes[t] = self.lower[t]

            if (np.dot(votes, self.alpha) >= 0):
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

