import os

ALGORITHMS = ['lambda_means']
DATA_DIR = 'datasets'
OUTPUT_DIR = 'output'
DATASETS = ['easy', 'hard', 'bio', 'finance', 'iris', 'speech', 'vision']


def run_on_dataset(dataset, algorithm, cluster_lambda=0., number_of_clusters=2, clustering_training_iterations=10):
    """  Run a particular clustering algorithm on a particular dataset with a given configuration  """
    print('Training algorithm %s on dataset %s...' % (algorithm, dataset))
    data = os.path.join(DATA_DIR, '%s.train' % dataset)
    model_file = os.path.join(OUTPUT_DIR, '%s.train.%s.pkl' % (dataset, algorithm))
    unformatted_cmd = 'python3 classify.py --data %s --mode train --model-file %s --algorithm %s '
    unformatted_cmd += '--cluster_lambda %s --number_of_clusters %s --clustering_training_iterations %s'
    cmd = unformatted_cmd % (data, model_file, algorithm, cluster_lambda, number_of_clusters,
                             clustering_training_iterations)
    os.system(cmd)
    for subset in ['train', 'dev', 'test']:
        data = os.path.join(DATA_DIR, '%s.%s' % (dataset, subset))
        # Some datasets might not contain full train, dev, test splits; in this case we should continue without error:
        if not os.path.exists(data):
            continue
        print('Generating %s predictions on dataset %s (%s)...' % (algorithm, dataset, subset))
        model_file = os.path.join(OUTPUT_DIR, '%s.train.%s.pkl' % (dataset, algorithm))
        predictions_file = os.path.join(OUTPUT_DIR, '%s.%s.%s.predictions' % (dataset, subset, algorithm))
        unformatted_cmd = 'python3 clustering_python_scripts/cluster_accuracy.py --data %s --mode test --model-file %s --predictions-file %s'
        cmd = unformatted_cmd % (data, model_file, predictions_file)
        os.system(cmd)
        '''
        if subset != 'test':
            print('Computing accuracy obtained by %s on dataset %s (%s)...' % (algorithm, dataset, subset))
            cmd = 'python3 clustering_python_scripts/number_clusters.py %s' % (predictions_file)
            os.system(cmd)
        '''


if __name__ == "__main__":
    """  
    Run all both algorithms on all data sets with some configuration.  Experimentation is encouraged, particularly
    with the cluster_lambda parameter.  How should one choose the proper cluster_lambda to get the true K?
    """
    if not os.path.exists(DATA_DIR):
        raise Exception('Data directory specified by DATA_DIR does not exist.')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for algorithm1 in ALGORITHMS:
        for dataset1 in DATASETS:
            if dataset1 == 'iris':
                run_on_dataset(dataset1, algorithm1, number_of_clusters=3)
            else:
                run_on_dataset(dataset1, algorithm1, number_of_clusters=2)
