## Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, shutil

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from support_functions import unzip_file, moving_folder

## Prepare data
data_dir = "./data"
checkpoints_dir = "./checkpoints"
for file in os.listdir(data_dir):
    if file.endswith(".zip"):
        unzip_file(path=os.path.join(data_dir,file), des=data_dir, delete=True)
        

## Load checkpoint file
import pickle

with open(os.path.join(checkpoints_dir,"scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
