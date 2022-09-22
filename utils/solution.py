import csv
import numpy as np
from pathlib import Path

from .data_load import load_test

ROOT_PATH = Path('/home/ayden/dev/kaggle3181/')
DATA_PATH = ROOT_PATH.joinpath('data')
SOLUTIONS_PATH = ROOT_PATH.joinpath('solutions/')

TEST_DATA = load_test()

def create_solution(model, name = "my_solution", batch_size = 32):
    """Creates solution csv in solutions/
    Args:
        model (keras model), capable of calling .predict
    """
    preds = model.predict(TEST_DATA, batch_size=batch_size)

    categorical_labels = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'fish', 'horse', 'lion', 'spider']
    num_predicted_labels = np.argmax(preds, axis=1)
    cat_predicted_labels = [categorical_labels[num_predicted_labels[i]] for i in range(len(num_predicted_labels))]

    header= ['ID', 'Label']
    with open(SOLUTIONS_PATH.joinpath(f'{name}.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        data = []
        for i in range(len(cat_predicted_labels)):
            data.append([str(i), cat_predicted_labels[i]])
        writer.writerows(data)