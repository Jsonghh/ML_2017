import csv
import data as data
import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
import math
from sklearn.model_selection import train_test_split

# Read Data
data = pd.read_csv('/Users/jhe18/Desktop/ML_NTU_2017Fall/Open_Source_Projects/winequality-red.csv', sep=';')
print(data.head())

# Split X, Y, training and test data
Y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                       test_size=0.2,
                                                       random_state=123,
                                                       stratify=Y)
print(X_train[1][1])