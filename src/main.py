import pandas as pd
import numpy as np

from helper1 import clean_age, impute, clean_submitted, remove_data_anomalies
from helper2 import transform_us, transform_countries, remove_location_anomalies
from helper3 import join, join_helper

# disable chained assignments
pd.options.mode.chained_assignment = None 

train = pd.read_csv("data/cases_train.csv")
test = pd.read_csv("data/cases_test.csv")
location = pd.read_csv("data/location.csv")

# train_processed  = pd.read_csv("results/cases_train_processed.csv")
# test_processed  = pd.read_csv("results/cases_test_processed.csv")
# location_transformed = pd.read_csv("results/location_transformed.csv")

# countries_transformed.to_csv("results/location_transformed2.csv", index=False)

print("Training Data Proccessing Start")
# training data's ages have been fully cleaned
train_processed = clean_age(train)
# train data has been fully imputed
train_processed = impute(train_processed)
# train data has been cleaned of anomalies
train_processed = remove_data_anomalies(train_processed)

# cleaned training data has been written to "results/cases_train_processed.csv"
# train_processed.to_csv("results/cases_train_processed.csv", index=False)
print("Training Data Proccessing Finished\n")

print("Test Data Proccessing Start")
# test data's ages have been fully cleaned
test_processed = clean_age(test)
# test data's date_confirmation have been fully cleaned
test_processed = clean_submitted(test)
# test data has been fully imputed
test_processed = impute(test_processed)
# cleaned test data has been written to "results/cases_test_processed.csv"
# test_processed.to_csv("results/cases_test_processed.csv", index=False)
print("Test Data Proccessing Finished\n")

print("Remove Anomalies Start")
location_removed = remove_location_anomalies(location)
print("Remove Anomalies Finished\n")
print("Locations US Transform Start")
location_transformed = transform_us(location_removed)
location_transformed.to_csv("results/location_transformed.csv", index=False)
print("Locations US Transform Finished\n")
# extra transformation on location only for EDA purposes
print("Locations Countries Transform Start")
countries_transformed = transform_countries(location_removed)
print("Locations Countries Transform Finished\n")

print("CSV Join Start")
train_joined = join(train_processed, join_helper(location_removed), countries_transformed)
train_joined.to_csv("results/cases_train_processed.csv", index=False)
test_joined = join(test_processed, join_helper(location_removed), countries_transformed)
test_joined.to_csv("results/cases_test_processed.csv", index=False)
print("CSV Join Finished")