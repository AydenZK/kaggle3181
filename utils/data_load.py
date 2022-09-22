import os
from pathlib import Path

import numpy as np
from imutils import paths
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

from .models import AnimalsDatasetManager, SimplePreprocessor

"""Global Variables"""
ROOT_PATH = Path('/home/ayden/dev/kaggle3181/')
DATA_PATH = ROOT_PATH.joinpath('data')
TRAIN_PATH = DATA_PATH.joinpath('train_dataset/datasets/Animals/')
TEST_PATH = DATA_PATH.joinpath('test_dataset/official_test_aug/')

"""Local Functions"""
def create_label_folder_dict(adir):
    sub_folders= [folder for folder in os.listdir(adir)
                  if os.path.isdir(os.path.join(adir, folder))]
    label_folder_dict= dict()
    for folder in sub_folders:
        item= {folder: os.path.abspath(os.path.join(adir, folder))}
        label_folder_dict.update(item)
    return label_folder_dict

def load_test_set(folder):
    image_paths = sorted(list(paths.list_images(folder)))
    test_data = []
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert("RGB") #load_img(image_path)
        img = img.resize((32, 32), Image.ANTIALIAS)
        x = img_to_array(img)
        test_data.append(x)
        if i+1 % 500 ==0:
            print("Loaded {} images".format(i+1))
    test_data = np.asarray(test_data)
    return test_data

def load_train():
    """
    Returns:
        data_manager (AnimalsDatasetManager): data manager object
            containing training, validation and testing sets.
            see models.py
    """
    label_folder_dict = create_label_folder_dict(TRAIN_PATH)
    sp = SimplePreprocessor(width=32, height=32)
    data_manager = AnimalsDatasetManager([sp])
    data_manager.load(label_folder_dict, verbose=100)
    data_manager.process_data_label()
    data_manager.train_valid_test_split()

    assert len(data_manager.X_train) > 0
    print("----SUCCESS: LOADED DATASET----")

    return data_manager

def load_test() -> np.ndarray:
    """Loads test data from test path"""
    test_data = load_test_set(TEST_PATH)

    assert len(test_data) > 0
    print("----SUCCESS: LOADED DATASET----")

    return test_data
