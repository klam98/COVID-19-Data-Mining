import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("data/cases_train_processed.csv")

encoder = LabelEncoder()
data = data.apply(encoder.fit_transform)

x = data.iloc[:, data.columns != "outcome"] #input
y = data.iloc[:, data.columns == "outcome"] #output
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0, shuffle = False, stratify = None)

def randomforests_model(x_train, x_test, y_train, y_test):
	# define and train the model
    model = RandomForestClassifier(n_estimators=25)
    model.fit(x_train, y_train)

    # prediction on test set using the model
    y_predict = model.predict(x_test)

    # report performance
    print('Random Forests accuracy:', accuracy_score(y_test, y_predict))
    with open("models/rf_classifier.pkl", "wb") as file:
	    pickle.dump(model, file)

randomforests_model(x_train, x_test, y_train, y_test)