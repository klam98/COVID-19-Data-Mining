import pandas as pd
import numpy as np
import math

def transform(dataset):
	united_states = dataset[dataset["Country_Region"] == "US"]
	united_states["Incidence_Rate"].fillna(0, inplace = True)
	states = pd.DataFrame(united_states["Province_State"].unique()).values
	transformed  = pd.DataFrame(columns = dataset.columns)
	for state in states:
		latitude = longitude = confirmed = deaths = recovered = active = incidence_rate = count = 0
		for i in range(united_states.shape[0]):
			county = united_states.iloc[i]
			if county["Province_State"] == state:
				count += 1
				confirmed += county["Confirmed"]
				deaths += county["Deaths"]
				recovered += county["Confirmed"] - county["Active"] - county["Deaths"]
				active += county["Active"]
				incidence_rate += county["Incidence_Rate"]
				if not math.isnan(county["Lat"]):
					latitude += county["Lat"]
				if not math.isnan(county["Long_"]):
					longitude += county["Long_"]

		incidence_rate /= count
		case_fatality = deaths/confirmed*100
		latitude /= count
		longitude /= count
		transformed = transformed.append(pd.DataFrame({"Province_State": state, "Country_Region": "US", "Last_Update": "WIP", "Lat": latitude, "Long_": longitude, "Confirmed": confirmed, "Deaths": deaths, "Recovered": recovered,\
		 "Active": active, "Combined_Key": (state + ", US"), "Incidence_Rate": incidence_rate, "Case-Fatality_Ratio": case_fatality}), ignore_index=True)

	recovered_index = transformed[transformed["Province_State"] == "Recovered"].index
	transformed.drop(recovered_index, inplace = True)
	return transformed
