import pandas as pd
import numpy as np

from helper1 import clean_age, impute
from helper2 import transform_us, transform_countries
from helper3 import join, join_helper

# disable chained assignments
pd.options.mode.chained_assignment = None 

train = pd.read_csv("data/cases_train.csv")
test = pd.read_csv("data/cases_test.csv")
location = pd.read_csv("data/location.csv")

# train_processed  = pd.read_csv("results/cases_train_processed.csv")
# test_processed  = pd.read_csv("results/cases_test_processed.csv")
# location_transformed = pd.read_csv("results/location_transformed.csv")

location_transformed = transform_us(location)
location_transformed.to_csv("results/location_transformed.csv", index=False)

# extra transformation on location only for EDA purposes
countries_transformed = transform_countries(location)
# countries_transformed.to_csv("results/location_transformed2.csv", index=False)


# training data's ages have been fully cleaned
train_processed = clean_age(train)

# train data has been fully imputed
train_processed = impute(train_processed)

# cleaned training data has been written to "results/cases_train_processed.csv"
train_processed.to_csv("results/cases_train_processed.csv", index=False)



# test data's ages have been fully cleaned
test_processed = clean_age(test)

# test data has been fully imputed
test_processed = impute(test_processed)

# cleaned test data has been written to "results/cases_test_processed.csv"
test_processed.to_csv("results/cases_test_processed.csv", index=False)


train_joined = join(train_processed, join_helper(location), countries_transformed)
train_joined.to_csv("results/cases_train_joined.csv", index=False)

test_joined = join(test_processed, join_helper(location), countries_transformed)
test_joined.to_csv("results/cases_test_joined.csv", index=False)