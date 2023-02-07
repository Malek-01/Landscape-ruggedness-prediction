# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:04:13 2020

@author: Malek
"""

import numpy
import matplotlib.pyplot as plt
import pandas
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
numpy.random.seed(7)


# load the dataset
X = pandas.read_excel('TSP_instances_merged2.xlsx', usecols="A:B")
y = pandas.read_excel('TSP_instances_merged2.xlsx', usecols="J")
X = X.values
X = X.astype('float32')
y = y.values
y = y.astype('float32')

start_time = time.time()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# split into train and test sets
#train_size = int(len(dataset) * 0.05)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(len(train), len(test))

train_pct_index = int(0.75 * len(X))
train_pct_index2 = int(0.75 * len(y))

X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index2], y[train_pct_index:]

# reshape input to be [samples, time steps, features]

#trainX= trainX.reshape(3999,1)
#trainY= trainY.reshape(3999)
#testY= testY.reshape(1000)
#testX = testX.reshape(1000,1)

clf = GradientBoostingRegressor(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred = clf.predict(X_test)
y_pred2 = clf.predict(X_train)
print('R:', r2_score(y_test, y_pred)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('result')
print('R:', r2_score(y_train, y_pred2)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred2))  
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred2)) 
l = list(range(len(y_pred)))

plt.figure(figsize=(10, 8))
plt.plot(l, y_test, 'b-', label = 'Real values')
plt.plot(l, y_pred, 'r-', label = 'SVM predictions')
plt.xlabel('Sample'); plt.ylabel('ACF value'); plt.title('Prediction of landscape ruggedness for TSP using SVM (test set)')
plt.legend();

# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
#print('R:', r2_score(testY, testPredict)) 

# shift train predictions for plotting
#trainPredictPlot = numpy.empty_like(dataset)
#trainPredictPlot[:, :] = numpy.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
#testPredictPlot = numpy.empty_like(dataset)
#testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()


#fig, ax = plt.subplots()
#ax.scatter(trainX, testPredict)
#ax.plot(testY.max(), testY.max(), 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()