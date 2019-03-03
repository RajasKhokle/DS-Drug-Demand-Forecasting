# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:33:45 2019
@author: Rajas Khokle
Purpose: Create LSTM model for Time series modelling of the healthcare data
"""
# import libraries
import pandas as pd
import numpy as np 
import sqlalchemy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 


apple_training_complete = pd.read_csv(r'E:\Datasets\apple_training.csv')  
apple_training_processed = apple_training_complete.iloc[:, 1:2].value
scaler = MinMaxScaler(feature_range = (0, 1))
apple_training_scaled = scaler.fit_transform(apple_training_processed) 

features_set = []  
labels = []  
for i in range(60, 1260):  
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels) 
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  

model = Sequential()  
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2)) 
model.add(Dense(units = 1))  

model.compile(optimizer = 'adam', loss = 'mean_squared_error')  
model.fit(features_set, labels, epochs = 100, batch_size = 32) 

apple_testing_complete = pd.read_csv(r'E:\Datasets\apple_testing.csv')  
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values 


