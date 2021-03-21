import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
import pickle

data = pd.read_csv("data/cases_train_processed.csv")

encoder = LabelEncoder()
data = data.apply(encoder.fit_transform)

x = data.iloc[:, data.columns != "outcome"] #input
y = data.iloc[:, data.columns == "outcome"] #output
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0, shuffle = False, stratify = None)

def xgboost_model(x_train, y_train):
	model = xgboost.XGBClassifier()
	model.fit(x_train, y_train.values.ravel())
	with open("models/xgb_classifier.pkl", "wb") as file:
		pickle.dump(model, file)
	return model

def knn_model(x_train, y_train):
	model = neighbors.KNeighborsClassifier(100, weights='distance')
	model.fit(x_train, y_train.values.ravel())
	with open("models/knn_classifier.pkl", "wb") as file:
		pickle.dump(model, file)
	return model

def randomforests_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=25)
    model.fit(x_train, y_train.values.ravel())
    with open("models/rf_classifier.pkl", "wb") as file:
	    pickle.dump(model, file)
    return model

def accuracy(model, x, y):
	y_predict = model.predict(x)
	accuracy = accuracy_score(y, y_predict)
	return accuracy

# xgboost_model(x_train, y_train)
# knn_model(x_train, y_train)
# randomforests_model(x_train, y_train)

loaded_xgboost = pickle.load(open("models/xgb_classifier.pkl", "rb"))
loaded_knn = pickle.load(open("models/knn_classifier.pkl", "rb"))
loaded_rf = pickle.load(open("models/rf_classifier.pkl", "rb"))

print("XGBoost Training  Accuracy: ", accuracy(loaded_xgboost, x_train, y_train))
print("K-Nearest Neighbours Training  Accuracy: ", accuracy(loaded_knn, x_train, y_train))
print("Random Forests Training  Accuracy: ", accuracy(loaded_rf, x_train, y_train))

print("XGBoost Validation Accuracy: ", accuracy(loaded_xgboost, x_test, y_test))
print("K-Nearest Neighbours Validation Accuracy: ", accuracy(loaded_knn, x_test, y_test))
print("Random Forests Validation Accuracy: ", accuracy(loaded_rf, x_test, y_test))