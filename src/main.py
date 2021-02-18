import pandas as pd
import numpy as np

from helper1 import clean_age, impute

train = pd.read_csv("data/cases_train.csv")
test = pd.read_csv("data/cases_test.csv")


#prints rows in test data with any missing values in meaningful columns
#--NEEDS TO BE IMPUTED(?)--
train_missing = train.iloc[:, :7]
train_missing = train_missing[pd.isna(train_missing).any(1)]
# print(train_missing)

train_processed = clean_age(train)
train_processed = impute(train_processed)
print(train_processed)
#training data's ages have been fully cleaned




#prints rows in test data with any missing values in meaningful columns
#--NEEDS TO BE IMPUTED(?)--
test_missing = test.iloc[:, :7]
test_missing = test_missing[pd.isna(test_missing).any(1)]
# print(test_missing)

#prints rows with an age value that is non-NaN but not numeric
#--NEEDS TO BE CLEANED--
test_a = test.iloc[:, :7]
test_a  = test_a.dropna()
a = pd.to_numeric(test_a["age"], errors='coerce').isna()
test_non_numeric = test_a[a]
# print(test_non_numeric)

test_processed = clean_age(test)

#prints rows that still have non-numeric values after cleaning
test_processed = test_processed.iloc[:, :7]
test_processed  = test_processed.dropna()
b = pd.to_numeric(test_processed["age"], errors='coerce').isna()
test_non_numeric2 = test_processed[b]
# print(test_non_numeric2)
# print(test_processed)