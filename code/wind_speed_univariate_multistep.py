#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[2]:


def load_data(url):
  df = pd.read_csv(url, sep = "\t", header = None)
  df.columns = ['id','date','longitude','latitude','speed']
  df = df.drop(['date'], axis = 1)
  df['category'] = df['speed'].apply(lambda x: 
  1 if x<=27 else 2  if x<=33 and x> 27 else 3 if x<=47 and x> 33 else 4 if x<=63 and x> 47 else 5 if x<=89 and x> 63 else 6 if x<=119 and x>89 else 7 )
  return df


# In[3]:


url = 'https://raw.githubusercontent.com/sydney-machine-learning/cyclone_deeplearning/main/data/SouthIndianOcean/misc/rawtrain1985-2001.txt'
df = load_data(url)
speed = df['speed'].tolist()

train_south_pacific = 5474
train_south_indian = 8628

# In[4]:


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
    return np.sqrt(((pred-actual) ** 2).mean())
  
def categorical(pred, actual):
  cm = confusion_matrix(pred,actual)
  ps = precision_score(pred,actual,average='micro')
  rs = recall_score(pred,actual,average='micro')
  f1 = f1_score(pred,actual, average = 'micro')
  return cm,ps,rs,f1


# In[5]:


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


# In[6]:


univariate = True # if false, its multivariate case
n_steps_in = 6
n_seq = 2
n_steps_out = 1
n_features = 1 # for univariate
Hidden = 10
Epochs = 50
Num_Exp = 5


# In[7]:


def vanilla(hidden,n_steps_in,n_steps_out,n_features):
    model = Sequential()
    model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features), dropout=0.2))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    return model


# In[8]:


def bidirectional(hidden,n_steps_in,n_steps_out,n_features):
  model = Sequential()
  model.add(Bidirectional(LSTM(Hidden, activation='relu'), input_shape=(n_steps_in, n_features)))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  return model


# In[9]:


def cnn_lstm(hidden,n_steps_in,n_steps_out,n_features,n_seq):
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, int(n_steps_in/n_seq), n_features)))
  model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(Hidden, activation='relu'))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  return model


# In[10]:


def conv_lstm(hidden,n_steps_in,n_steps_out,n_features,n_seq):
  model = Sequential()
  model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, int(n_steps_in/n_seq), n_features)))
  model.add(Flatten())
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  return model


# In[11]:


#all models
def MODEL_LSTM(model_name, univariate, x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden):

    train_acc = np.zeros(Num_Exp)
    test_acc = np.zeros(Num_Exp)

    if model_name == 'vanilla':
      model = vanilla(Hidden,n_steps_in,n_steps_out,n_features)
    elif model_name == 'bidirectional':
      model = bidirectional(Hidden,n_steps_in,n_steps_out,n_features)
    elif model_name == 'cnn-lstm':
      model = cnn_lstm(Hidden,n_steps_in,n_steps_out,n_features,n_seq)
    elif model_name == 'conv-lstm':
      model = conv_lstm(Hidden,n_steps_in,n_steps_out,n_features, n_seq)
    
    #model.summary()

    y_predicttest_allruns = np.zeros([Num_Exp, x_test.shape[0], y_test.shape[1]])

    #print(y_predicttest_allruns.shape, ' shape ')


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
        #print(train_acc[run], 'train accuracy')
        test_acc[run] = rmse(y_predicttest, y_test)
        if test_acc[run] < Best_RMSE:
            Best_RMSE = test_acc[run]
            Best_Predict_Test = y_predicttest
        
    train_std = np.std(train_acc)
    test_std = np.std(test_acc)
    print("Total time for", Num_Exp, "experiments", time.time() - start_time)
    return train_acc, test_acc, train_std, test_std, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns


# In[12]:


models = ['vanilla', 'bidirectional', 'cnn-lstm', 'conv-lstm']
models


# In[13]:


mean_train = {
    'vanilla':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]},
    'bidirectional':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}, 
    'cnn-lstm':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]},
    'conv-lstm':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]},
}

mean_test = {
    'vanilla':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]},
    'bidirectional':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}, 
    'cnn-lstm':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]},
    'conv-lstm':{'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]},
}

categories = {
    'vanilla':{'cm_train': 1,'ps_train': 1,'rs_train': 1,'f1_train': 1,'cm_test': 1,'ps_test': 1, 'rs_test': 1, 'f1_test': 1},
    'bidirectional':{'cm_train': 1,'ps_train': 1,'rs_train': 1,'f1_train': 1,'cm_test': 1,'ps_test': 1, 'rs_test': 1, 'f1_test': 1},
    'cnn-lstm':{'cm_train': 1,'ps_train': 1,'rs_train': 1,'f1_train': 1,'cm_test': 1,'ps_test': 1, 'rs_test': 1, 'f1_test': 1},
    'conv-lstm':{'cm_train': 1,'ps_train': 1,'rs_train': 1,'f1_train': 1,'cm_test': 1,'ps_test': 1, 'rs_test': 1, 'f1_test': 1},
}


# In[18]:


for j in range(6):
    n_steps_out = j+1
    print('---------------------------------------------------------')
    print('no of steps out: ', n_steps_out)

    for i in models:
        print("for " + i + ":")

        if i == 'vanilla' or i=='bidirectional':
            train = speed[:train_south_indian]
            test = speed[train_south_indian+1:]
            x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
            x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
            y_train = y_train.reshape((y_train.shape[0],n_steps_out))
            y_test = y_test.reshape((y_test.shape[0],n_steps_out))
        elif i == 'cnn-lstm':
            train = speed[:train_south_indian]
            test = speed[train_south_indian+1:]
            x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
            x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
            x_train = x_train.reshape((x_train.shape[0], n_seq, int(n_steps_in/n_seq), n_features))
            x_test = x_test.reshape((x_test.shape[0], n_seq, int(n_steps_in/n_seq), n_features))
        elif i=='conv-lstm':
            train = speed[:train_south_indian]
            test = speed[train_south_indian+1:]
            x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
            x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
            x_train = x_train.reshape((x_train.shape[0], n_seq, 1, int(n_steps_in/n_seq), n_features))
            x_test = x_test.reshape((x_test.shape[0], n_seq, 1, int(n_steps_in/n_seq), n_features))
        
        #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        train_acc, test_acc, train_std_dev, test_std_dev, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns = MODEL_LSTM(i, univariate,x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs, Hidden)
        mean_train[i][str(j+1)].append(np.mean(train_acc, axis=0))
        mean_test[i][str(j+1)].append(np.mean(test_acc, axis=0))
        mean_train[i][str(j+1)].append(train_std_dev)
        mean_test[i][str(j+1)].append(test_std_dev)

        y_predicttrain = y_predicttrain.tolist()
        if j==0:
          actual_cat_train=cat_calc(y_train)
          predicted_cat_train=cat_calc(y_predicttrain)
          actual_cat_test=cat_calc(y_test)
          predicted_cat_test=cat_calc(Best_Predict_Test.tolist())
          categories[i]['cm_train'], categories[i]['ps_train'], categories[i]['rs_train'], categories[i]['f1_train'] = categorical(predicted_cat_train,actual_cat_train)
          categories[i]['cm_test'], categories[i]['ps_test'], categories[i]['rs_test'], categories[i]['f1_test'] = categorical(predicted_cat_test,actual_cat_test)

        print(mean_train[i], 'mean rmse, std train') 
        print(mean_test[i], 'mean rmse, std test')
        #print(categories[i])


with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([mean_train, mean_test, categories], f)

# Getting back the objects:
with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    mean_train, mean_test, categories = pickle.load(f)

    
print('computations done')

# In[26]:
barWidth = 0.2
fig = plt.subplots(figsize =(12, 8))

vanilla = [mean_test['vanilla'][str(i+1)][0] for i in range(6)]
bidirectional = [mean_test['bidirectional'][str(i+1)][0] for i in range(6)]
cnn_lstm = [mean_test['cnn-lstm'][str(i+1)][0] for i in range(6)]
conv_lstm = [mean_test['conv-lstm'][str(i+1)][0] for i in range(6)]

yer1 = [mean_test['vanilla'][str(i+1)][1] for i in range(6)]
yer2 = [mean_test['bidirectional'][str(i+1)][1] for i in range(6)]
yer3 = [mean_test['cnn-lstm'][str(i+1)][1] for i in range(6)]
yer4 = [mean_test['conv-lstm'][str(i+1)][1] for i in range(6)]

# Set position of bar on X axis
br1 = np.arange(len(vanilla))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]


# Make the plot
plt.bar(br1, vanilla, yerr = yer1, color ='r', width = barWidth,
        edgecolor ='grey', label ='vanilla')
plt.bar(br2, bidirectional, yerr = yer2, color ='g', width = barWidth,
        edgecolor ='grey', label ='bidirectional')
plt.bar(br3, cnn_lstm, yerr = yer3, color ='b', width = barWidth,
        edgecolor ='grey', label ='cnn-lstm')
plt.bar(br4, conv_lstm, yerr = yer4, color ='y', width = barWidth,
        edgecolor ='grey', label ='conv-lstm')
 
# Adding Xticks
plt.xlabel('Steps', fontweight ='bold', fontsize = 15)
plt.ylabel('RMSE', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(vanilla))],
        ['1', '2', '3', '4', '5', '6'], fontsize = 20)

plt.legend()
plt.savefig('result_multistep_south_indian.png')

df = pd.DataFrame()
df['model'] = models
for i in range(6):
    out = i+1
    df[str(out)+'-step-out-rmse'] = [mean_test[j][str(out)][0] for j in models]
    df[str(out)+'-step-out-std'] = [mean_test[j][str(out)][1] for j in models]
    
df.to_csv('result_south_indian.csv', index=False)

df = pd.DataFrame()
df['model'] = models

for i in models:
    df['precision_score_test'] = [categories[j]['ps_test'] for j in models]
    df['recall_score_test'] = [categories[j]['rs_test'] for j in models]
    df['f1_score_test'] = [categories[j]['f1_test'] for j in models]
    
df.to_csv('cat_pred_result_south_indian.csv', index=False)    

print('done')
