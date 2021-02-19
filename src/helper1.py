import pandas as pd
import numpy as np

def impute(dataset):
	#impute age column with mean age
	dataset["age"] = pd.to_numeric(dataset['age'],errors='coerce', downcast='signed')
	mean = dataset['age'].mean()
	dataset['age'].fillna(mean, inplace=True)
	dataset = dataset.astype({"age": int})

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
	dataset['latitude'].fillna("Unknown", inplace=True)
	dataset['longitude'].fillna("Unknown", inplace=True)
	dataset = dataset.astype({"latitude": float})
	dataset = dataset.astype({"longitude": float})

	#impute additional information column with "None"
	dataset['additional_information'].fillna("None", inplace=True)

	#impute source column with "Unknown"
	dataset['source'].fillna("Unknown", inplace=True)
	return dataset

def clean_age(dataset):
	#STILL MISSING DOB TO AGE PROCESSING
	processed = dataset.replace("0-1", "1", regex=True)
	processed = processed.replace("0-4", "2", regex=True)
	processed = processed.replace("00-04", "2", regex=True)
	processed = processed.replace("0-9", "5", regex=True)
	processed = processed.replace("0-10", "5", regex=True)
	processed = processed.replace("0-18", "9", regex=True)
	processed = processed.replace("0-19", "10", regex=True)
	processed = processed.replace("0-20", "10", regex=True)
	processed = processed.replace("0-60", "30", regex=True)

	processed = processed.replace("5-9", "7", regex=True)
	processed = processed.replace("5-14", "10", regex=True)
	processed = processed.replace("05-14", "10", regex=True)

	processed = processed.replace("10-14", "12", regex=True)
	processed = processed.replace("10-19", "15", regex=True)
	processed = processed.replace("11-80", "46", regex=True)
	processed = processed.replace("13-19", "16", regex=True)
	processed = processed.replace("13-69", "41", regex=True)
	processed = processed.replace("15-19", "17", regex=True)
	processed = processed.replace("15-34", "24", regex=True)
	processed = processed.replace("18-49", "34", regex=True)
	processed = processed.replace("18-50", "34", regex=True)
	processed = processed.replace("18-60", "39", regex=True)
	processed = processed.replace("18-65", "42", regex=True)
	processed = processed.replace("18-99", "59", regex=True)
	processed = processed.replace("19-65", "42", regex=True)
	processed = processed.replace("19-75", "47", regex=True)

	processed = processed.replace("20-24", "22", regex=True)
	processed = processed.replace("20-29", "25", regex=True)
	processed = processed.replace("20-30", "25", regex=True)
	processed = processed.replace("20-39", "30", regex=True)
	processed = processed.replace("20-69", "45", regex=True)
	processed = processed.replace("20-70", "45", regex=True)
	processed = processed.replace("21-61", "41", regex=True)
	processed = processed.replace("22-60", "41", regex=True)
	processed = processed.replace("22-23", "23", regex=True)
	processed = processed.replace("23-84", "54", regex=True)
	processed = processed.replace("25-29", "27", regex=True)
	processed = processed.replace("27-29", "28", regex=True)
	processed = processed.replace("27-40", "34", regex=True)
	processed = processed.replace("28-35", "32", regex=True)

	processed = processed.replace("30-34", "32", regex=True)
	processed = processed.replace("30-39", "35", regex=True)
	processed = processed.replace("30-70", "50", regex=True)
	processed = processed.replace("33-78", "56", regex=True)
	processed = processed.replace("35-39", "37", regex=True)
	processed = processed.replace("35-59", "47", regex=True)
	processed = processed.replace("36-45", "41", regex=True)
	processed = processed.replace("37-38", "38", regex=True)

	processed = processed.replace("40-49", "45", regex=True)
	processed = processed.replace("40-50", "45", regex=True)
	processed = processed.replace("41-60", "51", regex=True)
	processed = processed.replace("45-49", "47", regex=True)

	processed = processed.replace("50-54", "52", regex=True)
	processed = processed.replace("50-59", "55", regex=True)
	processed = processed.replace("50-69", "60", regex=True)
	processed = processed.replace("50-60", "55", regex=True)
	processed = processed.replace("50-100", "75", regex=True)
	processed = processed.replace("54-56", "55", regex=True)
	processed = processed.replace("55-59", "57", regex=True)

	processed = processed.replace("60-60", "60", regex=True)
	processed = processed.replace("60-64", "62", regex=True)
	processed = processed.replace("60-69", "65", regex=True)
	processed = processed.replace("60-70", "65", regex=True)
	processed = processed.replace("60-79", "69", regex=True)
	processed = processed.replace("65-69", "67", regex=True)

	processed = processed.replace("70-74", "72", regex=True)
	processed = processed.replace("70-79", "75", regex=True)
	processed = processed.replace("74-76", "75", regex=True)
	processed = processed.replace("75-79", "77", regex=True)

	processed = processed.replace("80-84", "82", regex=True)
	processed = processed.replace("80-89", "85", regex=True)

	processed = processed.replace("18 - 100", "59", regex=True)

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