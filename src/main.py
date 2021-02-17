import pandas as pd
import numpy as np

test = pd.read_csv("data/cases_test.csv")
train = pd.read_csv("data/cases_train.csv")

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
print(test_non_numeric)

test_processed = test.replace("0-4", "2", regex=True)
test_processed = test_processed.replace("00-04", "2", regex=True)
test_processed = test_processed.replace("0-9", "5", regex=True)
test_processed = test_processed.replace("0-18", "9", regex=True)
test_processed = test_processed.replace("15-34", "24", regex=True)
test_processed = test_processed.replace("20-29", "25", regex=True)
test_processed = test_processed.replace("30-39", "35", regex=True)
test_processed = test_processed.replace("35-59", "47", regex=True)
test_processed = test_processed.replace("40-49", "45", regex=True)
test_processed = test_processed.replace("50-59", "55", regex=True)
test_processed = test_processed.replace("60-69", "65", regex=True)
test_processed = test_processed.replace("60-79", "69", regex=True)
test_processed = test_processed.replace("70-79", "75", regex=True)
test_processed = test_processed.replace("80-89", "85", regex=True)
test_processed = test_processed.replace("80+", "80", regex=False)
test_processed = test_processed.replace("90+", "90", regex=False)



test_processed = test_processed.iloc[:, :7]
test_processed  = test_processed.dropna()
b = pd.to_numeric(test_processed["age"], errors='coerce').isna()
test_non_numeric2 = test_processed[b]
print(test_non_numeric2)
# print(test_processed)

