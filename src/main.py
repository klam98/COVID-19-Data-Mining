import numpy as np 
import pandas as pd
import sklearn as sk
import sklearn.model_selection


data = pd.read_csv("data/cases_train_processed.csv")
train, test = sk.model_selection.train_test_split(data.head(5), train_size = 0.8, random_state = 0, shuffle = False, stratify = None)
