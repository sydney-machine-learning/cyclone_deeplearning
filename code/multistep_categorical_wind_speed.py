# -*- coding: utf-8 -*-

import copy
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
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
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set_theme() 
import errno
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
import pickle
from sklearn.metrics import roc_curve, auc
from scipy import interp

def load_data(url):
  df = pd.read_csv(url, sep = "\t", header = None)
  df.columns = ['id','date','longitude','latitude','speed']
  df = df.drop(['date'], axis = 1)
  df['category'] = df['speed'].apply(lambda x: 
  1 if x<=27 else 2  if x<=33 and x> 27 else 3 if x<=47 and x> 33 else 4 if x<=63 and x> 47 else 5 if x<=89 and x> 63 else 6 if x<=119 and x>89 else 7 )
  return df

ocean = 'south_pacific'  #south_indian or south_pacific
if ocean == 'south_indian':
    url = 'https://raw.githubusercontent.com/sydney-machine-learning/cyclone_deeplearning/main/data/SouthIndianOcean/misc/rawtrain1985-2001.txt'
    train_limit = 8000
    hot_encoded_result_file_name = 'south_indian'
    category_result_file_name = 'roc_data_south_indian'
    
else:
    url = 'https://raw.githubusercontent.com/sydney-machine-learning/cyclone_deeplearning/main/data/SouthPacificOcean/misc/rawtrain1985-2005.txt'
    train_limit = 5000
    hot_encoded_result_file_name = 'south_pacific'
    category_result_file_name = 'roc_data_south_pacific' 
    
df = load_data(url)
speed = df['speed'].tolist()
categories = df['category'].tolist()

def split_sequence(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
def rmse(pred, actual):
    return np.sqrt(((pred-actual) ** 2).mean())
  
def categorical(pred, actual):
  cm = confusion_matrix(pred,actual)
  ps = precision_score(pred,actual,average='micro')
  rs = recall_score(pred,actual,average='micro')
  f1 = f1_score(pred,actual, average = 'micro')
  return cm,ps,rs,f1

def cat_calc(tsd): 
    output=np.empty(len(tsd))
    for i in range(len(tsd)):
        if tsd[i][0]<=27:
            output[i]=1
        elif tsd[i][0]>27 and tsd[i][0]<=33:
            output[i]=2
        elif tsd[i][0]>33 and tsd[i][0]<=47:
            output[i]=3
        elif tsd[i][0]>47 and tsd[i][0]<=63:
            output[i]=4
        elif tsd[i][0]>63 and tsd[i][0]<=89:
            output[i]=5
        elif tsd[i][0]>89 and tsd[i][0]<=119:
            output[i]=6
        else:
            output[i]=7
    return output

univariate = True # if false, its multivariate case
n_steps_in = 6
n_seq = 2
n_steps_out = 1
n_features = 8 # one hot encoding of category
Hidden = 10
Epochs = 50
Num_Exp = 3

def vanilla(n_steps_in,n_steps_out,n_features):
  model = Sequential()
  model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features)))
  model.add(RepeatVector(n_steps_out))
  model.add(LSTM(Hidden, activation='relu', return_sequences=True))
  model.add(TimeDistributed(Dense(n_features, activation = "softmax")))
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model

def bidirectional(n_steps_in,n_steps_out,n_features):
  model = Sequential()
  model.add(Bidirectional(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features))))
  model.add(RepeatVector(n_steps_out))
  model.add(Bidirectional(LSTM(Hidden, activation='relu', return_sequences=True)))
  model.add(TimeDistributed(Dense(n_features, activation = "softmax")))
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model

def cnn_lstm(n_steps_in,n_steps_out,n_features,n_seq):
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, int(n_steps_in/n_seq), n_features)))
  model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(Hidden, activation='relu'))
  model.add(RepeatVector(n_steps_out))
  model.add(LSTM(Hidden, activation='relu', return_sequences=True))
  model.add(TimeDistributed(Dense(n_features, activation = "softmax")))
  model.compile(optimizer='adam', loss='mse')
  return model

def conv_lstm(n_steps_in,n_steps_out,n_features,n_seq):
  model = Sequential()
  model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, int(n_steps_in/n_seq), n_features)))
  model.add(Flatten())
  model.add(RepeatVector(n_steps_out))
  model.add(LSTM(Hidden, activation='relu', return_sequences=True))
  model.add(TimeDistributed(Dense(n_features, activation = "softmax")))
  model.compile(optimizer='adam', loss='mse')
  return model

#all models
def MODEL_LSTM(model_name, univariate, x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden):

    train_acc = np.zeros(Num_Exp)
    test_acc = np.zeros(Num_Exp)

    if model_name == 'vanilla':
      model = vanilla(n_steps_in,n_steps_out,n_features)
    elif model_name == 'bidirectional':
      model = bidirectional(n_steps_in,n_steps_out,n_features)
    elif model_name == 'cnn-lstm':
      model = cnn_lstm(n_steps_in,n_steps_out,n_features,n_seq)
    elif model_name == 'conv-lstm':
      model = conv_lstm(n_steps_in,n_steps_out,n_features, n_seq)
    
    #model.summary()

    y_predicttest_allruns = np.zeros([Num_Exp, x_test.shape[0], y_test.shape[1]])

    #print(y_predicttest_allruns.shape, ' shape ')


    Best_RMSE = 1000  # Assigning a large number
    start_time = time.time()
    for run in range(Num_Exp):
        print("Experiment", run + 1, "in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Epochs, batch_size=10, verbose=0, shuffle=False)
        scores = model.predict_proba(x_test)
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        #y_predicttest_allruns[run,:,:] = y_predicttest
        train_acc[run] = rmse(y_predicttrain, y_train)
        #print(train_acc[run], 'train accuracy')
        test_acc[run] = rmse(y_predicttest, y_test)
        if test_acc[run] < Best_RMSE:
            Best_RMSE = test_acc[run]
            Best_Predict_Test = y_predicttest
        
    train_std = np.std(train_acc)
    test_std = np.std(test_acc)
    print("Total time for", Num_Exp, "experiments", time.time() - start_time)
    return train_acc, test_acc, train_std, test_std, Best_Predict_Test, y_predicttrain, y_predicttest, scores

models = ['vanilla', 'bidirectional', 'cnn-lstm', 'conv-lstm']

predictions_train = dict()
actual_train = dict()
predictions_test = dict()
actual_test = dict()

# one hot encode
categories = to_categorical(df['category'])

for j in range(6):
    predictions_train_per_step = dict()
    actual_train_per_step = dict()  
    predictions_test_per_step = dict()
    actual_test_per_step = dict()
    n_steps_out = j+1
    print('---------------------------------------------------------')
    print('no of steps out: ', n_steps_out)

    for i in models:
        print("for " + i + ":")

        if i == 'vanilla' or i=='bidirectional':
            train = categories[0:train_limit]
            test = categories[train_limit:]
            x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
            x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
        elif i == 'cnn-lstm':
            train = categories[0:train_limit]
            test = categories[train_limit:]
            x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
            x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
            x_train = x_train.reshape((x_train.shape[0], n_seq, int(n_steps_in/n_seq), n_features))
            x_test = x_test.reshape((x_test.shape[0], n_seq, int(n_steps_in/n_seq), n_features))
        elif i=='conv-lstm':
            train = categories[0:train_limit]
            test = categories[train_limit:]
            x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
            x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
            x_train = x_train.reshape((x_train.shape[0], n_seq, 1, int(n_steps_in/n_seq), n_features))
            x_test = x_test.reshape((x_test.shape[0], n_seq, 1, int(n_steps_in/n_seq), n_features))
        
        #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        train_acc, test_acc, train_std_dev, test_std_dev, Best_Predict_Test, y_predicttrain, y_predicttest, scores = MODEL_LSTM(i, univariate,x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs, Hidden)
        predictions_train_per_step[i] = y_predicttrain
        actual_train_per_step[i] = y_train
        predictions_test_per_step[i] = Best_Predict_Test
        actual_test_per_step[i] = y_test
        
    predictions_train[str(j+1)] = predictions_train_per_step 
    actual_train[str(j+1)] = actual_train_per_step
    predictions_test[str(j+1)] = predictions_test_per_step 
    actual_test[str(j+1)] = actual_test_per_step


with open(hot_encoded_result_file_name + '.pkl', 'wb') as f: 
    pickle.dump([predictions_train, actual_train, predictions_test, actual_test], f)

# Getting back the objects:
with open(hot_encoded_result_file_name + '.pkl', 'rb') as f: 
    predictions_train, actual_train, predictions_test, actual_test = pickle.load(f)
    
print('pickle done')

predictions_test_copy = predictions_test
actual_test_copy = actual_test
actual_label_test = dict()
pred_label_test = dict()
for j in range(6):
  n_steps_out = j+1
  model_act = dict()
  model_pred = dict()
  for i in models:
    act_each = []
    pred_each = []
    for k in range(actual_test_copy[str(n_steps_out)][i].shape[0]):
      act_list = []
      pred_list = []
      for l in range(n_steps_out):
        tmp = actual_test_copy[str(n_steps_out)][i][k][l]
        tmp2 = predictions_test_copy[str(n_steps_out)][i][k][l]        
        index = tmp.argmax()
        index2 = tmp2.argmax()
        act_list.append(index)
        pred_list.append(index2)
      act_each.append(act_list)
      pred_each.append(pred_list)
    model_act[i] = act_each
    model_pred[i] = pred_each
  actual_label_test[str(n_steps_out)] = model_act 
  pred_label_test[str(n_steps_out)] = model_pred

predictions_train_copy = predictions_train
actual_train_copy = actual_train
actual_label_train = dict()
pred_label_train = dict()
for j in range(6):
  n_steps_out = j+1
  model_act = dict()
  model_pred = dict()
  for i in models:
    act_each = []
    pred_each = []
    for k in range(actual_train_copy[str(n_steps_out)][i].shape[0]):
      act_list = []
      pred_list = []
      for l in range(n_steps_out):
        tmp = actual_train_copy[str(n_steps_out)][i][k][l]
        tmp2 = predictions_train_copy[str(n_steps_out)][i][k][l]        
        index = tmp.argmax()
        index2 = tmp2.argmax()
        act_list.append(index)
        pred_list.append(index2)
      act_each.append(act_list)
      pred_each.append(pred_list)
    model_act[i] = act_each
    model_pred[i] = pred_each
  actual_label_train[str(n_steps_out)] = model_act 
  pred_label_train[str(n_steps_out)] = model_pred

    
metrics_train = dict()
metrics_test = dict()


for j in range(6):
  n_steps_out = j+1
  model_metric = dict()
  for i in models:
    step_metric = dict()
    for k in range(j+1):
        act = [actual_label_train[str(n_steps_out)][i][m][k] for m in range(len(actual_label_train[str(n_steps_out)][i]))]
        pred = [pred_label_train[str(n_steps_out)][i][m][k] for m in range(len(pred_label_train[str(n_steps_out)][i]))]
        step_metric[str(k+1)] = categorical(np.asarray(act), np.asarray(pred))
    model_metric[i] = step_metric
  metrics_train[str(n_steps_out)] = model_metric 

for j in range(6):
  n_steps_out = j+1
  model_metric = dict()
  for i in models:
    step_metric = dict()
    for k in range(j+1):
        act = [actual_label_test[str(n_steps_out)][i][m][k] for m in range(len(actual_label_test[str(n_steps_out)][i]))]
        pred = [pred_label_test[str(n_steps_out)][i][m][k] for m in range(len(pred_label_test[str(n_steps_out)][i]))]
        step_metric[str(k+1)] = categorical(np.asarray(act), np.asarray(pred))
    model_metric[i] = step_metric
  metrics_test[str(n_steps_out)] = model_metric 


n_classes = 8 # number of class
# Compute ROC curve and ROC area for each class

roc_data_test = dict()
for j in range(6):
  n_steps_out = j+1
  model_roc = dict()
  for i in models:
    step_roc = dict()
    for l in range(j+1):
        data = dict()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        act = [actual_test[str(n_steps_out)][i][m][l] for m in range(actual_test[str(n_steps_out)][i].shape[0])]
        pre = [predictions_test[str(n_steps_out)][i][m][l] for m in range(predictions_test[str(n_steps_out)][i].shape[0])]
        act = np.stack( act, axis=0 )
        pre = np.stack( pre, axis=0 )
        for k in range(n_classes):
            fpr[str(k)], tpr[str(k)], _ = roc_curve(act[:, k], pre[:, k])
            roc_auc[str(k)] = auc(fpr[str(k)], tpr[str(k)])
        fpr["micro"], tpr["micro"], _ = roc_curve(act.ravel(), pre.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr = np.unique(np.concatenate([fpr[str(p)] for p in range(1,n_classes)]))

# Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for p in range(1,n_classes):
            mean_tpr += interp(all_fpr, fpr[str(p)], tpr[str(p)])

# Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        data['fpr'] = fpr
        data['tpr'] = tpr
        data['roc_auc'] = roc_auc
        step_roc[str(l+1)] = data
    model_roc[i] = step_roc
  roc_data_test[str(n_steps_out)] = model_roc

n_classes = 8 # number of class
# Compute ROC curve and ROC area for each class

roc_data_train = dict()
for j in range(6):
  n_steps_out = j+1
  model_roc = dict()
  for i in models:
    step_roc = dict()
    for l in range(j+1):
        data = dict()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        act = [actual_train[str(n_steps_out)][i][m][l] for m in range(actual_train[str(n_steps_out)][i].shape[0])]
        pre = [predictions_train[str(n_steps_out)][i][m][l] for m in range(predictions_train[str(n_steps_out)][i].shape[0])]
        act = np.stack( act, axis=0 )
        pre = np.stack( pre, axis=0 )
        for k in range(n_classes):
            fpr[str(k)], tpr[str(k)], _ = roc_curve(act[:, k], pre[:, k])
            roc_auc[str(k)] = auc(fpr[str(k)], tpr[str(k)])
        fpr["micro"], tpr["micro"], _ = roc_curve(act.ravel(), pre.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr = np.unique(np.concatenate([fpr[str(p)] for p in range(1,n_classes)]))

# Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for p in range(1,n_classes):
            mean_tpr += interp(all_fpr, fpr[str(p)], tpr[str(p)])

# Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        data['fpr'] = fpr
        data['tpr'] = tpr
        data['roc_auc'] = roc_auc
        step_roc[str(l+1)] = data
    model_roc[i] = step_roc
  roc_data_train[str(n_steps_out)] = model_roc

with open(category_result_file_name + '.pkl', 'wb') as f: 
    pickle.dump([metrics_train, metrics_test, roc_data_train, roc_data_test], f)
    
    
index = [1.1, 2.1,2.2,3.1,3.2,3.3,4.1,4.2,4.3,4.4,5.1,5.2,5.3,5.4,5.5,6.1,6.2,6.3,6.4,6.5,6.6]
df = pd.DataFrame(columns=['vanilla-train', 'vanilla-test', 'bidirectional-train','bidirectional-test',
                          'cnn-lstm-train', 'cnn-lstm-test', 'conv-lstm-train', 'conv-lstm-test'])

rows = []

for j in range(1,7):
  for k in range(1,j+1):
        row = []
        for i in models:
            row.append(roc_data_train[str(j)][i][str(k)]['roc_auc']['macro'])
            row.append(roc_data_test[str(j)][i][str(k)]['roc_auc']['macro'])
        rows.append(row)
        
for i in range(len(rows)):
    df.loc[i] = rows[i]
df['steps'] = index
df.set_index(['steps'], inplace=True)

df.to_csv('roc_' + ocean + '.csv')
print('done')
