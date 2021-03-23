import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  
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
	model = xgboost.XGBClassifier(use_label_encoder = False, eval_metric="mlogloss") #default is max_depth = 6
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

def report(model, x, y):
	y_predict = model.predict(x)
	target_names = ['recovered', 'hospitalized', 'nonhospitalized', 'deceased']
	report = classification_report(y, y_predict, target_names=target_names, digits=4)
	return report

def xgboost_plot(x_train, y_train, x_test, y_test):
	train_scores, test_scores = list(), list()
	values = [i for i in range(1, 35)]
	for i in values:
	    # configure the model
	    model = xgboost.XGBClassifier(use_label_encoder = False, max_depth = i, eval_metric="mlogloss")
	    # fit model on the training dataset
	    model.fit(x_train, y_train)
	    # evaluate on the train dataset
	    train_y_predict = model.predict(x_train)
	    train_acc = accuracy_score(y_train, train_y_predict)
	    train_scores.append(train_acc)
	    # evaluate on the test dataset
	    test_y_predict = model.predict(x_test)
	    test_acc = accuracy_score(y_test, test_y_predict)
	    test_scores.append(test_acc)
	    print(i)
	# plot of train and test scores vs tree depth
	plt.plot(values, train_scores, '-o', label='Train')
	plt.plot(values, test_scores, '-o', label='Validation')
	plt.title('XGBoost Max Depth Vs Accuracy')
	plt.xlabel('Max Depth')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('plots/xgboost_max_depth.pdf')
	# plt.show()

def knn_plot(x_train, y_train, x_test, y_test):
	train_scores, test_scores = list(), list()
	values = [i*5 for i in range(1, 19)]
	for i in values:
		model = neighbors.KNeighborsClassifier(i, weights='distance')
		model.fit(x_train, y_train.values.ravel())

		# evaluate on the train dataset
		train_y_predict = model.predict(x_train)
		train_acc = accuracy_score(y_train, train_y_predict)
		train_scores.append(train_acc)

		# evaluate on the test dataset
		test_y_predict = model.predict(x_test)
		test_acc = accuracy_score(y_test, test_y_predict)
		test_scores.append(test_acc)

		# summarize progress
		print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

	plt.plot(values, train_scores, '-o', label='Train')
	plt.plot(values, test_scores, '-o', label='Validation')
	plt.title('K-Nearest Neighbors (# of Neighbors Vs Accuracy)')
	plt.xlabel('# of Neighbors')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('plots/knn_max_neighbors.pdf')
	# plt.show()

def randomforests_plot(x_train, y_train, x_test, y_test):
	train_scores, test_scores = list(), list()
	values = [i for i in range(1, 30)]
	for i in values:
		model = RandomForestClassifier(max_depth=i)
		model.fit(x_train, y_train.values.ravel())

		# evaluate on the train dataset
		train_y_predict = model.predict(x_train)
		train_acc = accuracy_score(y_train, train_y_predict)
		train_scores.append(train_acc)

		# evaluate on the test dataset
		test_y_predict = model.predict(x_test)
		test_acc = accuracy_score(y_test, test_y_predict)
		test_scores.append(test_acc)

		# summarize progress
		print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

	plt.plot(values, train_scores, '-o', label='Train')
	plt.plot(values, test_scores, '-o', label='Validation')
	plt.title('Random Forests Max Depth Vs Accuracy')
	plt.xlabel('Max Depth')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('plots/rf_max_depth.pdf')
	# plt.show()
	
xgboost_plot(x_train, y_train, x_test, y_test)
knn_plot(x_train, y_train, x_test, y_test)
randomforests_plot(x_train, y_train, x_test, y_test)

loaded_xgboost = pickle.load(open("models/xgb_classifier.pkl", "rb"))
loaded_knn = pickle.load(open("models/knn_classifier.pkl", "rb"))
loaded_rf = pickle.load(open("models/rf_classifier.pkl", "rb"))

# print("XGBoost Training Accuracy: ", accuracy(loaded_xgboost, x_train, y_train))
# print("XGBoost Validation Accuracy: ", accuracy(loaded_xgboost, x_test, y_test))
print("XGBoost Validation Classification Report:\n", report(loaded_xgboost, x_test, y_test))

# print("K-Nearest Neighbours Training Accuracy: ", accuracy(loaded_knn, x_train, y_train))
# print("K-Nearest Neighbours Validation Accuracy: ", accuracy(loaded_knn, x_test, y_test))
print("K-Nearest Neighbours Validation Classification Report:\n", report(loaded_knn, x_test, y_test))

# print("Random Forests Training Accuracy: ", accuracy(loaded_rf, x_train, y_train))
# print("Random Forests Validation Accuracy: ", accuracy(loaded_rf, x_test, y_test))
print("Random Forests Validation Classification Report:\n", report(loaded_rf, x_test, y_test))