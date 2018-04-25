import os
import argparse
import pickle
import numpy as np

import models
from data import load_data


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    parser.add_argument("--data", type=str, required=True, help="The data file to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create (for training) or load (for testing).")
    parser.add_argument("--algorithm", type=str,
                        help="The name of the algorithm to use. (Only used for training; inferred from the model " +
                             "file at test time.)")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create. (Only used for testing.)")
    parser.add_argument("--cluster_lambda", type=float,
                        help="The value of lambda in lambda-means", default=0.0)
    parser.add_argument("--clustering_training_iterations", type=int,
                        help="The number of training EM iterations")
    parser.add_argument("--number_of_clusters", type=int,
                        help="The number of clusters (K) to be used.")
    args = parser.parse_args()
    return args


def check_args(args):
    mandatory_args = {'data', 'mode', 'model_file', 'algorithm', 'predictions_file', 'cluster_lambda',
                      'clustering_training_iterations', 'number_of_clusters'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception('Arguments that we provided are now renamed or missing. If you hand this in, ' +
                        'you will not get full credit')

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--model should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def main():
    args = get_args()
    check_args(args)

    if args.mode.lower() == "train":
        # Load the training data.
        X, y = load_data(args.data)

        # Create and train the model.
        if args.algorithm.lower() == 'useless':
            model = models.Useless()
            model.fit(X, y)
        elif args.algorithm.lower() == 'lambda_means':
            model = models.LambdaMeans()
            model.fit(X, y, lambda0=args.cluster_lambda, iterations=args.clustering_training_iterations)
        elif args.algorithm.lower() == 'stochastic_k_means':
            model = models.StochasticKMeans()
            model.fit(X, y, num_clusters=args.number_of_clusters, iterations=args.clustering_training_iterations)
        else:
            raise Exception('The model given by --model is not yet supported.')

        # Save the model.
        try:
            with open(args.model_file, 'wb') as f:
                pickle.dump(model, f)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping model pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        X, y = load_data(args.data)

        # Load the model.
        try:
            with open(args.model_file, 'rb') as f:
                model = pickle.load(f)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading model pickle.")

        # Compute and save the predictions.
        y_hat = model.predict(X)
        np.savetxt(args.predictions_file, y_hat, fmt='%d')
            
    else:
        raise Exception("Mode given by --mode is unrecognized.")


if __name__ == "__main__":
    main()