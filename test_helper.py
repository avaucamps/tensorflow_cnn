import os
from pathlib import Path
import numpy as np
import collections
import pandas as pd

def number_of_testing_images(path):
    return len([name for name in os.listdir(path)])


def get_labels(train_path):
    label_names = sorted(item.name for item in Path(train_path).iterdir() if item.is_dir())
    label_names = [label for index, label in enumerate(label_names)]

    return label_names


def get_ids(test_data_path):
    ids = []
    for i in Path(test_data_path).iterdir():
        filename = os.path.basename(str(i))
        ids.append(filename)

    return ids

def get_predictions_label(predictions, labels):
    label_predictions = []
    for i in predictions:
        max_index = np.argmax(i)
        label_name = labels[max_index]
        label_predictions.append(label_name)

    return label_predictions    


def create_submission_file(predictions, filename, labels, test_data_path):
    data = collections.OrderedDict()
    data['file'] = get_ids(test_data_path)
    data['species'] = get_predictions_label(predictions, labels)
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(filename, index=False)