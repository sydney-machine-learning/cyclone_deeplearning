# -*- coding: utf-8 -*-
"""wind_speed.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g_vgUhs9v6pp2qgLP0s77SiwfY0vsv99
"""

import copy
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import hstack
from sklearn.preprocessing import StandardScaler
import datetime
import time
import joblib
from datetime import timedelta, date
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import os
import seaborn as sns; sns.set_theme() 
import errno

url = "https://raw.githubusercontent.com/rohitash-chandra/CMTL_dynamictimeseries/master/IndianOcean/rawtrain1985-2001.txt"
df = pd.read_csv(url, sep = "\t", header = None)
df.columns = ['id','date','longitude','latitude','speed']
df = df.drop(['date'], axis = 1)
df.to_csv('adjusted.csv')
df

#making starting point of every cyclone same, spatial bias removed
id = 1
x0 = df['longitude'][0]
y0 = df['latitude'][0]
df['longitude'][0] = 0
df['latitude'][0] = 0
for i in range(1, df.shape[0]):
  if df['id'][i] == id :
    df['longitude'][i] = df['longitude'][i] - x0
    df['latitude'][i] = df['latitude'][i] - y0
  else:
    x0 = df['longitude'][i]
    y0 = df['latitude'][i]
    df['longitude'][i] = 0
    df['latitude'][i] = 0
    id = df['id'][i]

longi = array(df['longitude'])
lat = array(df['latitude'])
speed = array(df['speed'])
#univariate, so only working with speed
speed = speed.reshape((len(speed), 1))
speed.shape, speed

sc = StandardScaler()
speed_scaled = sc.fit_transform(speed)
speed_scaled

#for univariate
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
def rmse(pred, actual):
	return np.sqrt(((pred - actual) ** 2).mean())

univariate = True # if false, its multivariate case
n_steps_in = 3
n_steps_out = 1
n_features = 1 # for univariate
Hidden = 10
Epochs = 50
Num_Exp = 2

if univariate is True:  
  train = speed_scaled[0:8628]
  test = speed_scaled[8629:9364]
  x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
  x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
  x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
  print(x_train.shape)
  x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
  print(x_test.shape)
  y_train = y_train.reshape((y_train.shape[0],1))
  print(y_train.shape)
  y_test = y_test.reshape((y_test.shape[0],1))
  print(y_test.shape)

#yet to be done
#else:

model = Sequential()

def MODEL_LSTM(univariate, x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden):

	train_acc = np.zeros(Num_Exp)
	test_acc = np.zeros(Num_Exp)
 
	model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features), dropout=0.2))
	# model.add(Dense(Hidden)) (No particular need of this)
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	model.summary()

	y_predicttest_allruns = np.zeros([Num_Exp, x_test.shape[0], x_test.shape[1]])

	print(y_predicttest_allruns.shape, ' shape ')


	Best_RMSE = 1000  # Assigning a large number

	start_time = time.time()
	for run in range(Num_Exp):
		print("Experiment", run + 1, "in progress")
		# fit model
		model.fit(x_train, y_train, epochs=Epochs, batch_size=10, verbose=0, shuffle=False)
		y_predicttrain = model.predict(x_train)
		y_predicttest = model.predict(x_test)
		y_predicttest_allruns[run,:,:] = y_predicttest
		train_acc[run] = rmse(y_predicttrain, y_train)
		print(train_acc[run], 'train accuracy')
		test_acc[run] = rmse(y_predicttest, y_test)
		if test_acc[run] < Best_RMSE:
			Best_RMSE = test_acc[run]
			Best_Predict_Test = y_predicttest

	print("Total time for", Num_Exp, "experiments", time.time() - start_time)
	return train_acc, test_acc, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns

train_acc, test_acc,Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns = MODEL_LSTM(univariate,x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs, Hidden)

print(train_acc, test_acc) 
mean_train = np.mean(train_acc, axis=0)
mean_test = np.mean(test_acc, axis=0)
print(mean_train, 'mean rmse train') 
print(mean_test, 'mean rmse test')

# test on unseen data
url_test = "https://raw.githubusercontent.com/rohitash-chandra/CMTL_dynamictimeseries/master/IndianOcean/rawtest2006-2013.txt"
df_test = pd.read_csv(url, sep = "\t", header = None)
df_test.columns = ['id','date','longitude','latitude','speed']
df_test = df_test.drop(['date'], axis = 1)
df_test.to_csv('adjusted_test.csv')
df_test

actual = array(df_test['speed'])
actual = actual.reshape((len(actual), 1))
actual_scaled = sc.fit_transform(actual)
X_TEST, Y_TEST = split_sequence(actual_scaled, n_steps_in, n_steps_out)
Y_TEST = Y_TEST.reshape(Y_TEST.shape[0],1)
predictions = model.predict(X_TEST)
rmse_score = rmse(predictions, Y_TEST)
print('rmse score on raw test data', rmse_score)
