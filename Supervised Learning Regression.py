# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:39:33 2024

@author: deji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")
from ydata_profiling import ProfileReport
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

data = pd.read_csv("5G_energy_consumption_dataset.csv")
data
data["Time"] = pd.to_datetime(data["Time"])
info = data.info()

missing = data.isnull().sum()

description = data.describe()

# report = ProfileReport(data, title = "5G energy consumption")
# report.to_file("5G energy consumption")

# correlation = data.corr()
# Energy is highly overall correlated with TXpower
# TXpower is highly overall correlated with Energy
# load is highly overall correlated with Energy	
# ESMODE has 87475 (94.4%) zeros

data = data.drop(["BS","Time"],axis = 1)

# standardize the data
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(data)
scaled_dataset = pd.DataFrame(scaled_dataset, columns = scaler.feature_names_in_)

# handling outliers
removed_outliers_data = scaled_dataset[(scaled_dataset >= -3) & (scaled_dataset <= 3)]
removed_outliers_data_total = removed_outliers_data.isnull().sum()

#handling missing values
imputer = SimpleImputer(strategy = "median")
new_data = imputer.fit_transform(removed_outliers_data)
new_data = pd.DataFrame(new_data, columns = imputer.feature_names_in_)

#selecting features and targets
y = new_data["Energy"]
x = new_data[["load","TXpower"]]

x_train,y_train,x_test,y_test = train_test_split(x,y,test_size = 0.20,random_state = 30)
model = LogisticRegression()
model.fit(x,y)
predicted = model.predict(x_test)
print("MSE: ",mean_squared_error(y_test,predicted))
print("R_squared:",metrics.r2_score(y_test,predicted))
















