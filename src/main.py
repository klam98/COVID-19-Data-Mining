import pandas as pd
import numpy as np

from helper1 import clean_age, impute
from helper2 import transform

# disable chained assignments
pd.options.mode.chained_assignment = None 

train = pd.read_csv("data/cases_train.csv")
test = pd.read_csv("data/cases_test.csv")
location = pd.read_csv("data/location.csv")

train_processed  = pd.read_csv("results/cases_train_processed.csv")
test_processed  = pd.read_csv("results/cases_test_processed.csv")

location_processed = transform(location)
location_processed.to_csv("results/location_processed.csv", index=False)
# training data's ages have been fully cleaned
# train_processed = clean_age(train)

# train data has been fully imputed
# train_processed = impute(train_processed)

# cleaned training data has been written to "results/cases_train_processed.csv"
# train_processed.to_csv("results/cases_train_processed.csv", index=False)



# test data's ages have been fully cleaned
# test_processed = clean_age(test)

# test data has been fully imputed
# test_processed = impute(test_processed)

# cleaned test data has been written to "results/cases_test_processed.csv"
# test_processed.to_csv("results/cases_test_processed.csv", index=False)
