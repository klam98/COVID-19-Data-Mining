import pandas as pd
import numpy as np

def transform(dataset):
	united_states = dataset[dataset["Country_Region"] == "US"]
	states = pd.DataFrame(united_states["Province_State"].unique()).values
	transformed  = pd.DataFrame(columns = dataset.columns)
	# dataset['Case-Fatality_Ratio'].fillna(0, inplace=True)
	for state in states:
		confirmed = deaths = recovered = active = combined = incidence_rate = case_fatality = 0
		# latitude, longitude = ""
		for i in range(0, united_states.shape[0]):
			county = united_states.iloc[i]
			if county["Province_State"] == state:
				confirmed += county["Confirmed"]
				deaths += county["Deaths"]
				recovered += county["Recovered"]
				active += county["Active"]
				incidence_rate += county["Incidence_Rate"]
				case_fatality += county["Case-Fatality_Ratio"]
		transformed = transformed.append(pd.DataFrame({"Province_State":state, "Country_Region":"US", "Last_Update":"WIP", "Lat":0.0, "Long_":0.0, "Confirmed": confirmed, "Deaths": deaths, "Recovered": recovered,\
		 "Active": active, "Combined_Key": combined, "Incidence_Rate": incidence_rate, "Case-Fatality_Ratio": case_fatality}), ignore_index=True)
	return transformed
