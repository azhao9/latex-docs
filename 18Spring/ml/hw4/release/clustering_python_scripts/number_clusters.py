import sys

if __name__ == "__main__":
    """  Get label distribution from a prediction file  """
    if len(sys.argv) != 2:
        print('usage: %s predictions' % sys.argv[0])
        sys.exit()
    predictions_file = sys.argv[1]
    predictions = open(predictions_file)
    # Load the labels:
    predicted_labels = []
    for line in predictions:
        predicted_labels.append(line.strip())
    predictions.close()
    labels = {}
    for l in predicted_labels:
        if l not in labels:
            labels[l] = 0
        labels[l] += 1
    print('Label distribution:')
    for l in labels:
        print('Label %s: %d' % (l, labels[l]))
    print('%d unique labels' % len(labels))
