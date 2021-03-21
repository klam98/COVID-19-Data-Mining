import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("data/cases_train_processed.csv")

encoder = LabelEncoder()
data = data.apply(encoder.fit_transform)

x = data.iloc[:, data.columns != "outcome"] #input
y = data.iloc[:, data.columns == "outcome"] #output
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0, shuffle = False, stratify = None)

def knn_model(x_train, x_test, y_train, y_test):
	model = neighbors.KNeighborsClassifier(100, weights='distance')
	model.fit(x_train,y_train)

	y_predict = model.predict(x_test)
	y_predict = [round(value) for value in y_predict]

	accuracy = accuracy_score(y_test, y_predict)
	print(accuracy) #0.887077237155495
	with open("models/knn_classifier.pkl", "wb") as file:
		pickle.dump(model, file)

knn_model(x_train, x_test, y_train, y_test)