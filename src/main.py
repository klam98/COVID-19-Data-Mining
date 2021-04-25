import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, plot_confusion_matrix, f1_score, recall_score
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
	outcomes = ['recovered', 'hospitalized', 'nonhospitalized', 'deceased']
	report = classification_report(y, y_predict, target_names=outcomes, digits=4)
	return report

def confusion_matrix_plot(model, x, y, title):
	outcomes = ['recovered', 'hospitalized', 'nonhospitalized', 'deceased']
	matrix = plot_confusion_matrix(model, x, y, display_labels=outcomes, xticks_rotation=15)
	matrix.ax_.set_title(title)
	plt.tight_layout()
	plt.show()

def cross_validation(model, x, y):
	scoring = {
		# labels=[0] is 'recovered'
        # labels=[1] is 'hospitalized'
        # labels=[2] is 'nonhospitalized'
        # labels=[3] is 'deceased'
		'F1-Score on deceased': make_scorer(f1_score, labels=[3], average=None),
		'Recall on deceased': make_scorer(recall_score, labels=[3], average=None),
		'Overall Accuracy': make_scorer(accuracy_score),
		'Overall Recall': make_scorer(recall_score, average='weighted')
	}
	gs = GridSearchCV(model, param_grid={'max_depth': range(2, 10, 2)},
				  scoring=scoring, refit='F1-Score on deceased', return_train_score=True)
	gs.fit(x, y)
	results = gs.cv_results_
	plt.figure(figsize=(13, 13))
	plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
			  fontsize=16)

	plt.xlabel("max_depth")
	plt.ylabel("Score")

	ax = plt.gca()
	ax.set_xlim(0, 10)
	ax.set_ylim(0.7, 1)

	# Get the regular numpy array from the MaskedArray
	X_axis = np.array(results['param_max_depth'].data, dtype=float)

	for scorer, color in zip(sorted(scoring), ['g', 'k']):
		for sample, style in (('train', '--'), ('test', '-')):
			sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
			sample_score_std = results['std_%s_%s' % (sample, scorer)]
			ax.fill_between(X_axis, sample_score_mean - sample_score_std,
							sample_score_mean + sample_score_std,
							alpha=0.1 if sample == 'test' else 0, color=color)
			ax.plot(X_axis, sample_score_mean, style, color=color,
					alpha=1 if sample == 'test' else 0.7,
					label="%s (%s)" % (scorer, sample))

		best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
		best_score = results['mean_test_%s' % scorer][best_index]

		# Plot a dotted vertical line at the best score for that scorer marked by x
		ax.plot([X_axis[best_index], ] * 2, [0, best_score],
				linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

		# Annotate the best score for that scorer
		ax.annotate("%0.2f" % best_score,
					(X_axis[best_index], best_score + 0.005))

	plt.legend(loc="best")
	plt.grid(False)
	plt.show()

# saved_xgboost = xgboost_model(x_train, y_train)
# saved_knn = knn_model(x_train, y_train)
# saved_rf = randomforests_model(x_train, y_train)

loaded_xgboost = pickle.load(open("models/xgb_classifier.pkl", "rb"))
loaded_knn = pickle.load(open("models/knn_classifier.pkl", "rb"))
loaded_rf = pickle.load(open("models/rf_classifier.pkl", "rb"))

# print("XGBoost Training Accuracy: ", accuracy(loaded_xgboost, x_train, y_train))
# print("XGBoost Validation Accuracy: ", accuracy(loaded_xgboost, x_test, y_test))
# print("XGBoost Training Classification Report:\n", report(loaded_xgboost, x_train, y_train))
# print("XGBoost Validation Classification Report:\n", report(loaded_xgboost, x_test, y_test))
# confusion_matrix_plot(loaded_xgboost, x_test, y_test, 'XGBoost Confusion Matrix')

# print("K-Nearest Neighbours Training Accuracy: ", accuracy(loaded_knn, x_train, y_train))
# print("K-Nearest Neighbours Validation Accuracy: ", accuracy(loaded_knn, x_test, y_test))
# print("K-Nearest Neighbours Training Classification Report:\n", report(loaded_knn, x_train, y_train))
# print("K-Nearest Neighbours Validation Classification Report:\n", report(loaded_knn, x_test, y_test))
# confusion_matrix_plot(loaded_rf, x_test, y_test, 'Random Forests Confusion Matrix')

# print("Random Forests Training Accuracy: ", accuracy(loaded_rf, x_train, y_train))
# print("Random Forests Validation Accuracy: ", accuracy(loaded_rf, x_test, y_test))
# print("Random Forests Training Classification Report:\n", report(loaded_rf, x_train, y_train))
# print("Random Forests Validation Classification Report:\n", report(loaded_rf, x_test, y_test))
# confusion_matrix_plot(loaded_knn, x_test, y_test, 'K-Nearest Neighbours Confusion Matrix')

cross_validation(loaded_xgboost, x_train, y_train)
# print(cross_validation(loaded_knn, x_train, y_train))
# print(cross_validation(loaded_rf, x_train, y_train))