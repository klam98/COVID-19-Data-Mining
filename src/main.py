import numpy as np 
import pandas as pd
import sklearn as sk
import sklearn.model_selection


data = pd.read_csv("data/cases_train_processed.csv").to_numpy()
Y = data[:, 9]
X = np.delete(data, 9, 1)

X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, train_size = 0.8, random_state = 0, shuffle = False, stratify = None)
