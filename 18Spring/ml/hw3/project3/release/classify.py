import os
import argparse
import sys
import pickle
import numpy as np

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

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
                    index = int(item.split(":")[0]) - 1
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
        self.j = np.array(num_boosting_iterations)
        self.c = np.array(num_boosting_iterations)
        self.alpha = np.array(num_boosting_iterations)

    def train(self, instances):
        N = len(instances)
        num_features = np.max([len(instance.feature_vector._feature_vector) for instance in instances])

        # Initialize uniform distribution
        D = np.ones(N)/N

        stop_hypothesis = self.num_boosting_iterations

        def h(i, j, c, instances):
            values = [instance.feature_vector.get(j) for instance in instances]
            y = [instance.label._label for instance in instances]
            x = instances[i].feature_vector._feature_vector

            if (x[j] > c):
                correct = np.where(values > c, 1, 0)
            else:
                correct = np.where(values <= c, 1, 0)

            result = np.dot(correct, y)
            if (result >= 0):
                return 1
            else:
                return -1

        for t in range(self.num_boosting_iterations):
            # Initialize error array along each feature
            error_j = [None] * N
            y = np.array([instance.label._label for instance in instances])

            for j in range(num_features):
                unique = np.sort(np.unique([instance.feature_vector.get(j) for instance in instances]))

                error_c = np.zeros(len(unique))

                for k in range(len(unique)):
                    for i in range(N):
                        if (h(i, j, unique[k], instances) != y[i]):
                            error_c += D[i]
            
                error_j[j] = error_c

            # Finding index of argmin j
            min_inds = [np.argmin(error_j[j]) for j in range(num_features)]  
            min_feature_ind = np.argmin([error_j[j][min_inds[j]] for j in range(num_features)])
            self.j[t] = min_feature_ind

            # Finding value of argmax c
            min_c_ind = np.argmin(error_j[min_feature_ind])
            self.c[t] = np.sort(np.unique(np.array([instance.feature_vector.get(min_feature_ind) for instance in instances])))[min_c_ind]

            error_t = 0
            for i in range(N):
                if(h(i, self.j[t], self.c[t], instances) != y[i]):
                    error_t += D[i]

            if (error_t < 0.000001):
                # stop and only use up to previous hypothesis
                stop_hypothesis = t
                continue

            self.alpha[t] = 0.5 * np.log((1-error_t)/error(t))

            # Calculate normalizing factor
            Z = 0
            for i in range(N):
                Z += D[i] * np.exp(-self.alpha[t] * y[i] * h(i, self.j[t], self.c[t], instances))

            # Calculate new distribution D
            for i in range(N):
                D[i] *= 1/Z * np.exp(-self.alpha[t] * y[i] * h(i, self.j[t], self.c[t], instances))

        # Truncate values if we stopped early, otherwise stop hypothesis is number of iterations
        self.j = self.j[:stop_hypothesis]
        self.c = self.c[:stop_hypothesis]
        self.alpha = self.alpha[:stop_hypothesis]

    def h(i, j, c, instances):
        values = np.array([instance.feature_vector.get(j) for instance in instances])
        y = np.array([instance.label._label for instance in instances])
        x = instances[i].feature_vector._feature_vector

        if (x[j] > c):
            correct = np.where(values > c, 1, 0)
        else:
            correct = np.where(values <= c, 1, 0)

        result = np.dot(correct, y)
        if (result >= 0):
            return 1
        else:
            return -1

    def predict(self, instance):
        # Sums for keeping track of whether y_hat = 1 or y_hat = -1 are better
        sum_1 = 0
        sum_2 = 0
        for t in range(len(self.alpha)):
            if (instance.feature_vector.get(self.j[t]) > self.c[t]):
                sum_1 += self.alpha[t]
            else:
                sum_2 += self.alpha[t]

        if (sum_1 >= sum_2):
            return 1
        else:
            return -1
       

def train(instances, args):
    if (args.algorithm.lower() == 'adaboost'):
        predictor = Adaboost(args.num_boosting_iterations)

    predictor.train(instances)

    # TODO This is where you will add new algorithms that will subclass Predictor
    
    return predictor


def write_predictions(predictor, instances, predictions_file):
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
        instances = load_data(args.data)

        # Train the model.
        predictor = train(instances, args)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        instances = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

