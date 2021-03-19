import numpy as np 
import pandas as pd
import sklearn as sk
import sklearn.model_selection


data = pd.read_csv("data/cases_train_processed.csv")
train, test = sk.model_selection.train_test_split(data, train_size = 0.8, random_state = 0)
train2, test2 = sk.model_selection.train_test_split(data, train_size = 0.8, random_state = 0)

print(train is train2)
print(test is test2)