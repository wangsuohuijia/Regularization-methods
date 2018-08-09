# -*- coding: UTF-8 -*-

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from keras import losses
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import LinearSVR


### Data preprocessing: transform the date to the numeric feature and save the update data
### as the "new_energydata_complete.csv"
# data = pd.read_csv('energydata_complete.csv')
# datecolumn = data['date']
# datelist = []
# for i in range(len(data)):
#     datelist.append(datecolumn[i])
# trans_datelist = []
# for j in datelist:
#     timeArray = time.strptime(j, "%Y-%m-%d %H:%M:%S")
#     timeStamp = int(time.mktime(timeArray))
#     trans_datelist.append(str(timeStamp))
# for m in range(len(data)):
#     data['date'] = data['date'].replace(data['date'][m], trans_datelist[m])
### write in a new csv
# data.to_csv('new_energydata_complete.csv')


### read the new data and shuffle it for the model
data = pd.read_csv('new_energydata_complete.csv')
data = data.drop(['Unnamed: 0'], axis=1)
index = [i for i in range(len(data))]
random.shuffle(index)
data = data.iloc[index, :]
y = data['Appliances']
x = data.drop(['Appliances'], axis=1)


### data normalization
x_norm_data = (x-x.min())/(x.max()-x.min())
y_norm_data = (y-y.min())/(y.max()-y.min())


### cut dataset, the cut proportion is 0.7:0.3.
X_train = x_norm_data.iloc[0:int(len(x_norm_data) * 0.7), :]
X_test = x_norm_data.iloc[int(len(x_norm_data) * 0.7):, :]
y_train = y_norm_data.iloc[0:int(len(y_norm_data) * 0.7),]
y_test = y_norm_data.iloc[int(len(y_norm_data) * 0.7):,]


# ########### SVM

clf = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
    kernel='linear', max_iter=10000, shrinking=True, tol=0.0001, verbose=False)
clf.fit(X_train, y_train)

prediction_svm = clf.predict(X_test)

coefficients_svm = clf.coef_
intercept_svm = clf.intercept_


### revert the prediction value
prediction_svm_ori = prediction_svm*(y.max()-y.min())+y.min()
y_test_ori = np.array(y_test*(y.max()-y.min())+y.min())

# calculate and plot residuals
residuals_svm = prediction_svm_ori - y_test_ori
plt.plot(residuals_svm)
plt.title('Residuals with SVM')
plt.ylabel('Residuals')
plt.show()

### calculate the mse value for the prediciton.
mse_svm = np.mean((prediction_svm_ori-y_test_ori)**2)
print('MSE with SVM:', mse_svm)


### plot the figure to see the difference between prediction and y_test.
plt.plot(y_test_ori, label='y_test_ori')
plt.plot(prediction_svm_ori, label='prediction_ori')
plt.title('Comparison between y_test and prediction with SVM (No penalty)')
plt.ylabel('Appliances')
plt.legend()
plt.show()


######## SVM with different penalties (L1 & L2)
### train the model:  we focus 3 parameter, C, loss and epsilon.
### for the loss: 'epsilon-insensitive loss' is for L1 and 'squared epsilon-insensitive loss' is for L2
### for L1, we can tune the epsilon value, for L2, we can tune the C value.
### others are the defaults.
regr = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='squared_epsilon_insensitive',
                 fit_intercept=True, intercept_scaling=1.0, dual=True,
                 verbose=0, random_state=None, max_iter=1000)

### fit the model
regr.fit(X_train, y_train)

### get the prediction
prediction_svm_p = regr.predict(X_test)

### revert the prediction value
prediction_svm_p_ori = prediction_svm_p*(y.max()-y.min())+y.min()
y_test_ori = np.array(y_test*(y.max()-y.min())+y.min())

### get the score for this model
# score = regr.score(X_test, y_test)

### calculate the mse value for the prediciton.
mse_svm_p = np.mean((prediction_svm_p_ori-y_test_ori)**2)
print("MSE with penalized SVM:", mse_svm_p)
### plot the figure to see the difference between prediction and y_test.
plt.plot(y_test_ori, label='y_test_ori')
plt.plot(prediction_svm_p_ori, label='prediction_ori')
plt.title('Comparison between y_test and prediction with SVM (L2 penalty)')
plt.ylabel('Applicances')
plt.legend()
plt.show()


########### Apply in MLP

### start to build MLP
model = Sequential()

### add the dense layer, 64 is the output dimension of this layer
### kernel_regularizer is the regularization of l1 and l2, we can choose any of them.
### 0.0001 is the λ
### input_shape is the dimension of the input data. we have 28 features, so it is 28.
model.add(Dense(64,
                kernel_regularizer=regularizers.l1_l2(0.0001),
                activation='sigmoid',
                input_shape=(28,)))

# model.add(Dense(64,
#                 activation='sigmoid',
#                 input_shape=(28,)))


### add another dense layer, it is the output layer, output a single value
### the activation function is linear, since we do the regression analysis.d
model.add(Dense(1, activation='linear'))

### compile the whole model together. loss function is mse, we use SGD to update the weights and biase.
### the lr is the learning rate, we choose 0.001.
### metrics is the 'mean_squared_error'
model.compile(loss=losses.mse,
              optimizer=SGD(lr=0.001),
              metrics=['mean_squared_error'])
### after creating the MLP, we can fit the model by using the training data.
### batch_size is for the SGD.
### epochs means every 10 epoch we record the result.
### verbose means we show the history of the model process.
hist = model.fit(x=X_train,
                 y=y_train,
                 batch_size=32,
                 epochs=10,
                 verbose=1)

### when we train the model, we can give the prediction for the test x
prediction_mlp = model.predict(X_test, batch_size=32)

### revert the prediction value
prediction_mlp_ori = prediction_mlp*(y.max()-y.min())+y.min()
y_test_ori = y_test*(y.max()-y.min())+y.min()


### we can evaluate data to see the model accuracy.
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)


### calculate the mse for the prediction
prediction_mlp_ori = prediction_mlp_ori.reshape(len(prediction_mlp_ori), )
# type(prediction)
y_test_ori = np.array(y_test_ori)
mlp_mse = np.mean((prediction_mlp_ori - y_test_ori) ** 2)
print(mlp_mse)


### plot the figure to observe the differece between y_test and prediction.
plt.plot(y_test_ori, label='y_test_ori')
plt.plot(prediction_mlp_ori, label='prediction_ori')
plt.title('Comparison between y_test and prediction with MLP (ElasticNet)')
plt.legend()
plt.show()
