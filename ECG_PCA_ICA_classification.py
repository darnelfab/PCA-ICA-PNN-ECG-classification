# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:09:37 2021

@author: darnel
"""

"""
ECG heartbeat classification using PCA/ICA dimensionality reduction
"""

#import and plot dataset
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
#import dataset
train_set=pd.read_csv('mitbih_database/113.csv', sep=',',header=0, parse_dates=[0],  squeeze=False)
X_train = train_set.iloc[:, 0:0]
Y_train = train_set.iloc[:, 1]
#plot dataset
plt.figure(figsize=(10,10))
plt.plot(Y_train.head(300),'r-', label='MLII train_set')
#plt.plot(df['V1'].head(300),'b-', label='V1')
plt.legend(loc="upper left")
plt.title("MLII train_set before Butterworth Fitler")
train_set.drop(train_set.index[:1], inplace=True)
#print(df.dtypes)
ytrain=train_set[['MLII']].head(300).stack().values.flatten()
#denoise ECG signal using Butterworth filter
# Butterworth filter
lowcut=0.01
highcut=15.0
signal_freq=360
filter_order=1
plt.figure(figsize=(10,10))
nyquist_freq = 0.5*signal_freq
low=lowcut/nyquist_freq
high=highcut/nyquist_freq
b, a = signal.butter(filter_order, [low,high], btype="band")
ytrain = signal.lfilter(b, a, ytrain)
plt.plot(ytrain)
plt.plot(Y_train.head(240),'r-', label='MLII')
plt.legend(loc="upper left")
plt.title("MLII train_set after Butterworth Fitler")

#test_set
test_set=pd.read_csv('mitbih_database/100.csv', sep=',',header=0, parse_dates=[0],  squeeze=False)
X_test = test_set.iloc[:, 0:0]
Y_test= test_set.iloc[:, 1]
#plot dataset
plt.figure(figsize=(10,10))
plt.plot(Y_test.head(300),'r-', label='MLII test_set')
#plt.plot(df['V1'].head(300),'b-', label='V1')
plt.legend(loc="upper left")
plt.title("MLII test_set Butterworth Fitler")
train_set.drop(train_set.index[:1], inplace=True)
#print(df.dtypes)
ytest=test_set[['MLII']].head(300).stack().values.flatten()
#denoise ECG signal using Butterworth filter
# Butterworth filter
lowcut=0.01
highcut=15.0
signal_freq=360
filter_order=1
plt.figure(figsize=(10,10))
nyquist_freq = 0.5*signal_freq
low=lowcut/nyquist_freq
high=highcut/nyquist_freq
b, a = signal.butter(filter_order, [low,high], btype="band")
ytest = signal.lfilter(b, a, ytest)
plt.plot(ytest)
plt.plot(Y_test.head(240),'r-', label='MLII')
plt.legend(loc="upper left")
plt.title("MLII test_set after Butterworth Fitler")

# performing preprocessing part on training and testing set such as fitting the Standard scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
ytrain_2d = ytrain.reshape(-1, 1)
ytrain = sc.fit_transform(ytrain_2d)
ytest_2d = ytest.reshape(-1, 1)
ytest = sc.transform(ytest_2d)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA

pca = PCA()
new_ytrain = pca.fit(ytrain_2d)
new_ytest = pca.fit(ytest_2d)
 
plt.plot(new_ytest)


  
