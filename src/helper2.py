import pandas as pd
import numpy as np
import math

# Preprocessing task 1.4
def transform_us(dataset):
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
		transformed = transformed.append(pd.DataFrame({"Province_State": state, "Country_Region": "United States", "Lat": latitude, "Long_": longitude, "Confirmed": confirmed, "Deaths": deaths, "Recovered": recovered,\
		 "Active": active, "Combined_Key": (state + ", US"), "Incidence_Rate": incidence_rate, "Case-Fatality_Ratio": case_fatality}), ignore_index=True)

	recovered_index = transformed[transformed["Province_State"] == "Recovered"].index
	transformed.drop(recovered_index, inplace = True)
	return transformed


# Additional function for aggregating location dataset by each country
def transform_countries(dataset):
	unique_countries = pd.DataFrame(dataset['Country_Region'].unique()).values
	transformed = pd.DataFrame(columns = ['Country_Region','Confirmed','Deaths','Recovered','Active','Incidence_Rate','Case-Fatality_Ratio']) 
	temp = dataset
	temp["Incidence_Rate"] = dataset["Incidence_Rate"].fillna(0)
	for country in unique_countries:
		confirmed = deaths = recovered = active = incidence_rate = count = 0
		for i in range(temp.shape[0]):
			region = temp.iloc[i]
			if region["Country_Region"] == country:
				count += 1
				confirmed += region["Confirmed"]
				deaths += region["Deaths"]
				recovered += region["Confirmed"] - region["Active"] - region["Deaths"]
				active += region["Active"]
				incidence_rate += region["Incidence_Rate"]

		incidence_rate /= count
		case_fatality = deaths / confirmed * 100
		transformed = transformed.append(pd.DataFrame({
        	"Country_Region": country, 
        	"Confirmed": confirmed, 
        	"Deaths": deaths, 
        	"Recovered": recovered,
        	"Active": active, 
        	"Incidence_Rate": incidence_rate, 
        	"Case-Fatality_Ratio": case_fatality
    	}), ignore_index=True)

	recovered_index = transformed[transformed["Country_Region"] == "Recovered"].index
	transformed.drop(recovered_index, inplace = True)
	return transformed