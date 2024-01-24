# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:39:33 2024

@author: busola
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from ydata_profiling import ProfileReport
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import math
from xgboost import XGBRegressor

data = pd.read_csv("5G_energy_consumption_dataset.csv")
data
data["Time"] = pd.to_datetime(data["Time"])
#extracting datetime features
data["Year"] = data["Time"].dt.year
data["Month"] = data["Time"].dt.month
data["Day"] = data["Time"].dt.day
data["Hour"] = data["Time"].dt.hour

#transforming BS column
data = pd.get_dummies(data,drop_first = True)

#turning time column to index
data = data.set_index("Time")

info = data.info()
unique = data.nunique()
missing = data.isnull().sum()

description = data.describe()
# report = ProfileReport(data, title = "5G energy consumption")
# report.to_file("5G energy consumption")

correlation = data.corr()
# Energy is highly overall correlated with TXpower
# TXpower is highly overall correlated with Energy
# load is highly overall correlated with Energy	
# ESMODE has 87475 (94.4%) zeros

# standardize the data
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(data)
scaled_dataset = pd.DataFrame(scaled_dataset, columns = scaler.feature_names_in_)

# # handling outliers
removed_outliers_data = scaled_dataset[(scaled_dataset >= -3) & (scaled_dataset <= 3)]
removed_outliers_data_total = removed_outliers_data.isnull().sum()

# #handling missing values
imputer = SimpleImputer(strategy = "median")
new_data = imputer.fit_transform(removed_outliers_data)
new_data = pd.DataFrame(new_data, columns = imputer.feature_names_in_)

# #selecting features and targets
y = new_data["Energy"]
x = new_data.drop(["Energy"], axis = 1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 30)
model = XGBRegressor(random_state = 0)
model.fit(x_train,y_train)
predicted = model.predict(x_test)
predictedtrain = model.predict(x_train)

mean_squared_errorstrain = mean_squared_error(y_train,predictedtrain)
print("RMSEtrain: ",math.sqrt(mean_squared_errorstrain))
print("R_squaredtrain:",metrics.r2_score(y_train,predictedtrain))

mean_squared_errors = mean_squared_error(y_test,predicted)
print("RMSE: ",math.sqrt(mean_squared_errors))
print("R_squared:",metrics.r2_score(y_test,predicted))

# RMSEtrain:  0.3718654860567501
# R_squaredtrain: 0.8429857090076217
# RMSE:  0.3899426029577845
# R_squared: 0.832465855236887


# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, predicted)

#u need to use a 3D scatterplot




# #r squared is like a 
# # was y changes explained by the changes in x

# # for a data frame to do a heatmap for only numerical or select only numerical
# # data.select_dtype("number")
# # for categorical, number becomes object






