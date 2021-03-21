import pandas as pd
import numpy as np

def impute(dataset):
	#impute age column with "Unknown"
	dataset["age"] = pd.to_numeric(dataset['age'],errors='coerce', downcast='signed')
	dataset['age'].fillna(-1, inplace=True)

	#impute sex column with "Unknown"
	dataset['sex'].fillna("Unknown", inplace=True)

	#impute province column with "Unknown"
	dataset['province'].fillna("Unknown", inplace=True)

	#impute country column with "Unknown", and fill in rows with Taiwan as a province with China as its country
	dataset['country'].fillna("Unknown", inplace=True)
	dataset['country'] = np.where(dataset['province'] == "Taiwan", "China", dataset['country'])

	#impute confirmation date column with "Unknown"
	dataset['date_confirmation'].fillna("Unknown", inplace=True)

	#impute latitude and longitude columns with "Unknown", and ensure type is float
	dataset["latitude"] = pd.to_numeric(dataset['latitude'],errors='coerce', downcast='signed')
	dataset["longtitude"] = pd.to_numeric(dataset['longitude'],errors='coerce', downcast='signed')
	dataset['latitude'].fillna(-91, inplace=True)
	dataset['longitude'].fillna(-181, inplace=True)
	# dataset = dataset.astype({"latitude": float})
	# dataset = dataset.astype({"longitude": float})

	#impute additional information column with "None"
	dataset['additional_information'].fillna("None", inplace=True)

	#impute source column with "Unknown"
	dataset['source'].fillna("Unknown", inplace=True)
	return dataset

def clean_age(dataset):
	#STILL MISSING DOB TO AGE PROCESSING
	processed = dataset.replace("0-1", "1", regex=False)
	processed = processed.replace("0-4", "2", regex=False)
	processed = processed.replace("00-04", "2", regex=False)
	processed = processed.replace("0-9", "5", regex=False)
	processed = processed.replace("0-10", "5", regex=False)
	processed = processed.replace("0-18", "9", regex=False)
	processed = processed.replace("0-19", "10", regex=False)
	processed = processed.replace("0-20", "10", regex=False)
	processed = processed.replace("0-60", "30", regex=False)

	processed = processed.replace("5-9", "7", regex=False)
	processed = processed.replace("5-14", "10", regex=False)
	processed = processed.replace("05-14", "10", regex=False)

	processed = processed.replace("10-14", "12", regex=False)
	processed = processed.replace("10-19", "15", regex=False)
	processed = processed.replace("11-80", "46", regex=False)
	processed = processed.replace("13-19", "16", regex=False)
	processed = processed.replace("13-69", "41", regex=False)
	processed = processed.replace("15-19", "17", regex=False)
	processed = processed.replace("15-34", "24", regex=False)
	processed = processed.replace("18-49", "34", regex=False)
	processed = processed.replace("18-50", "34", regex=False)
	processed = processed.replace("18-60", "39", regex=False)
	processed = processed.replace("18-65", "42", regex=False)
	processed = processed.replace("18-99", "59", regex=False)
	processed = processed.replace("19-65", "42", regex=False)
	processed = processed.replace("19-75", "47", regex=False)

	processed = processed.replace("20-24", "22", regex=False)
	processed = processed.replace("20-29", "25", regex=False)
	processed = processed.replace("20-30", "25", regex=False)
	processed = processed.replace("20-39", "30", regex=False)
	processed = processed.replace("20-69", "45", regex=False)
	processed = processed.replace("20-70", "45", regex=False)
	processed = processed.replace("21-61", "41", regex=False)
	processed = processed.replace("22-60", "41", regex=False)
	processed = processed.replace("22-23", "23", regex=False)
	processed = processed.replace("23-84", "54", regex=False)
	processed = processed.replace("25-29", "27", regex=False)
	processed = processed.replace("27-29", "28", regex=False)
	processed = processed.replace("27-40", "34", regex=False)
	processed = processed.replace("28-35", "32", regex=False)

	processed = processed.replace("30-34", "32", regex=False)
	processed = processed.replace("30-39", "35", regex=False)
	processed = processed.replace("30-70", "50", regex=False)
	processed = processed.replace("33-78", "56", regex=False)
	processed = processed.replace("35-39", "37", regex=False)
	processed = processed.replace("35-59", "47", regex=False)
	processed = processed.replace("36-45", "41", regex=False)
	processed = processed.replace("37-38", "38", regex=False)

	processed = processed.replace("40-49", "45", regex=False)
	processed = processed.replace("40-50", "45", regex=False)
	processed = processed.replace("41-60", "51", regex=False)
	processed = processed.replace("45-49", "47", regex=False)

	processed = processed.replace("50-54", "52", regex=False)
	processed = processed.replace("50-59", "55", regex=False)
	processed = processed.replace("50-69", "60", regex=False)
	processed = processed.replace("50-60", "55", regex=False)
	processed = processed.replace("50-100", "75", regex=False)
	processed = processed.replace("54-56", "55", regex=False)
	processed = processed.replace("55-59", "57", regex=False)

	processed = processed.replace("60-60", "60", regex=False)
	processed = processed.replace("60-64", "62", regex=False)
	processed = processed.replace("60-69", "65", regex=False)
	processed = processed.replace("60-70", "65", regex=False)
	processed = processed.replace("60-79", "69", regex=False)
	processed = processed.replace("65-69", "67", regex=False)

	processed = processed.replace("70-74", "72", regex=False)
	processed = processed.replace("70-79", "75", regex=False)
	processed = processed.replace("74-76", "75", regex=False)
	processed = processed.replace("75-79", "77", regex=False)

	processed = processed.replace("80-84", "82", regex=False)
	processed = processed.replace("80-89", "85", regex=False)

	processed = processed.replace("18 - 100", "59", regex=False)

	processed = processed.replace("18-", "18", regex=False)
	processed = processed.replace("65-", "65", regex=False)
	processed = processed.replace("80-", "80", regex=False)
	processed = processed.replace("80+", "80", regex=False)
	processed = processed.replace("85+", "85", regex=False)
	processed = processed.replace("90+", "90", regex=False)

	processed = processed.replace("5 month", "0", regex=False)
	processed = processed.replace("8 month", "1", regex=False)
	processed = processed.replace("11 month", "1", regex=False)

	return processed

def clean_submitted(dataset):
	processed = dataset.replace("10.03.2020 - 12.03.2020", "11.03.2020", regex=False)
	processed = processed.replace("10.03.2020-13.03.2020", "11.03.2020", regex=False)
	processed = processed.replace("12.03.2020 - 13.03.2020", "12.03.2020", regex=False)
	processed = processed.replace("25.02.2020 - 03.03.2020", "29.02.2020", regex=False)
	processed = processed.replace("07.03.2020 - 13.03.2020", "10.03.2020", regex=False)
	processed = processed.replace("07.03.2020 - 09.03.2020", "08.03.2020", regex=False)
	processed = processed.replace("18.03.2020-19.03.2020", "18.03.2020", regex=False)
	processed = processed.replace("07.03.2020 - 10.03.2020", "08.03.2020", regex=False)
	processed = processed.replace("05.03.2020-06.03.2020", "05.03.2020", regex=False)
	processed = processed.replace("10.03.2020 - 11.03.2020", "10.03.2020", regex=False)
	processed = processed.replace("25.02.2020 - 26.02.2020", "25.02.2020", regex=False)
	processed = processed.replace("12.03.2020-14.03.2020", "13.03.2020", regex=False)
	processed = processed.replace("07.03.2020-09.03.2020", "08.03.2020", regex=False)
	
	return processed

def remove_data_anomalies(dataset):
	index = dataset[(dataset.age == 0) & (dataset.province == "Lima")].index
	processed = dataset.drop(index)
	index = processed[(processed.age == 1) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 100) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 101) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 102) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 103) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 104) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 105) & (processed.province == "Lima")].index
	processed = processed.drop(index)
	index = processed[(processed.age == 106) & (processed.province == "Lima")].index
	processed = processed.drop(index)

	return processed