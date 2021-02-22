import pandas as pd
import numpy as np

from helper2 import transform_us


def join_helper(dataset):
	united_states = transform_us(dataset)
	other_countries = dataset[dataset["Country_Region"] != "US"]
	other_countries.drop(columns = "Last_Update", inplace = True)
	return pd.concat([united_states, other_countries])

def join_province(dataset_cases, dataset_loc):
	joined = pd.merge(dataset_cases, dataset_loc, how = "left", left_on = ["province", "country"], right_on = ["Province_State", "Country_Region"])
	# joined = pd.merge(dataset_cases, dataset_loc, how = "left", left_on = "country", right_on = "Country_Region")

	joined['Incidence_Rate'] = joined['Incidence_Rate'].apply(lambda x: x / 1000)

	# joined.drop(columns = ["Country_Region", "Confirmed", "Deaths", "Active", "Recovered", "Combined_Key", "Last_Update"], inplace = True)
	joined.drop(columns = ["Province_State", "Country_Region", "Lat", "Long_", "Confirmed", "Deaths", "Active", "Recovered", "Combined_Key", "Last_Update"], inplace = True)

	# joined.rename(columns={"Incidence_Rate": "Country_Incidence_Percentage", "Case-Fatality_Ratio": "Country_Case-Fatality_Ratio"}, inplace = True)
	joined.rename(columns={"Incidence_Rate": "Province_Incidence_Percentage", "Case-Fatality_Ratio": "Province_Case-Fatality_Ratio"}, inplace = True)
	return joined

def join_country(dataset_cases, dataset_loc):
	joined = pd.merge(dataset_cases, dataset_loc, how = "left", left_on = "country", right_on = "Country_Region")
	joined['Incidence_Rate'] = joined['Incidence_Rate'].apply(lambda x: x / 1000)
	joined.drop(columns = ["Country_Region", "Confirmed", "Deaths", "Active", "Recovered"], inplace = True)
	joined.rename(columns={"Incidence_Rate": "Country_Incidence_Percentage", "Case-Fatality_Ratio": "Country_Case-Fatality_Ratio"}, inplace = True)

	return joined

def join(dataset_cases, dataset_provinces, dataset_countries):
	return join_country(join_province(dataset_cases, dataset_provinces), dataset_countries)