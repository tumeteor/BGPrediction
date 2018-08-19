import sys
import numpy 
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from numpy import ma
from pykalman import KalmanFilter
#from sklearn.utils import check_arrays


def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)

    return numpy.mean(numpy.abs((y_true - y_pred) / (numpy.abs(y_true) + numpy.abs(y_pred)/2))) * 100


def hidenan(s):
   s = ma.array(s)
   for i in range(0,len(s)):
       if math.isnan(s[i]):
           s[i] = ma.masked 
   return s

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
      a = dataset[i:(i+look_back), 0]
      dataX.append(a)
      dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

csv = pd.read_csv(sys.argv[1],delimiter=";")
df = pd.DataFrame(csv)
df['date'] = pd.to_datetime(df['date'])
s = df.set_index('date')
s = s["2017-03-01 00:00:00":"2017-03-31"]
min_date = min(s.index)
print s
s = s.resample('H', 'sum')
s = s[min_date.replace(minute=0, second=0):"2017-03-31"]
kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance'])
if sys.argv[2] == "pad":
  print "performing padding interpolation"
  s_in = s.fillna(method="pad")
elif sys.argv[2]== "cubic":
  print "performing cubic spline interpolation"
  s_in = s.interpolate(method='cubic')
elif sys.argv[2] == "zero":
  print "performing fill 0"
  s_in = s.fillna(0)
elif sys.argv[2] == "kalma":
  print "performing kalma smoothing"
  dataset = s.values 
  dataset = hidenan(dataset)
  dataset = kf.em(dataset).smooth(dataset)[0]
  print dataset
else:
  print "performing linear interpolation"
  s_in = s.interpolate(method='linear')


#idx for actual values
idx = []
cnt = 1 
for index, row in s.iterrows(): 
  if not math.isnan(row[0]): 
     idx.append(cnt)
  cnt += 1

if sys.argv[2] != "kalma": dataset = s_in.values

#dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#print dataset
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#actual values
trainYa = []
for i in idx:
  # the indices are shifted twice here
  # recheck
  if i<2 or i > len(trainY): continue
  trainYa.append(trainY.item(i-2))
  
trainYa = numpy.array(trainYa)
print "actual train values:"
print trainYa
#actual values
testYa = []
for i in idx:
  if i <= len(trainY): continue
  if i - len(trainY) > len(testY): continue
  # because seems some values are missing in the middle
  # safer to do from the end back
  rest = len(dataset) - i 
  testYa.append(testY.item(len(testY)-rest))
print "actual test values:"
print testYa
testYa = numpy.array(testYa)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(SimpleRNN(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainYa =  scaler.inverse_transform([trainYa])
testYa =  scaler.inverse_transform([testYa])

trainYp =  []
for i in idx:
  if i<2 or i > len(trainPredict): continue
  trainYp.append(trainPredict.item(i-2))

testYp =  []
for i in idx:
  if i <= len(trainPredict): continue
  if i - len(trainPredict) > len(testY): continue
  rest = len(dataset) - i
  testYp.append(testPredict.item(len(testPredict)-rest))
 
trainYp = numpy.array(trainYp)
trainYp = scaler.inverse_transform(trainYp)

testYp = numpy.array(testYp)
testYp = scaler.inverse_transform(testYp)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error 
#print testYa[0]
#print testYp

print "all test values"
print testY
print "actual test values"
print testYa

a_trainMAE = mean_absolute_error(trainYa[0], trainYp)
a_trainRMSE =  math.sqrt(mean_squared_error(trainYa[0], trainYp))
a_trainSMAPE = mean_absolute_percentage_error(trainYa[0], trainYp)
print('Actual train Score: %.2f SMAPE %.2f RMSE  %.2f MAE' % (a_trainSMAPE, a_trainRMSE, a_trainMAE))
trainMAE = mean_squared_error(trainY[0], trainPredict[:,0])
trainRMSE = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainSMAPE = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])

print('Train Score:  %.2f SMAPE %.2f RMSE %.2f MAE'% (trainSMAPE, trainRMSE, trainMAE))
a_testMAE =  mean_absolute_error(testYa[0], testYp)
a_testRMSE =  math.sqrt(mean_squared_error(testYa[0], testYp))
a_testSMAPE = mean_absolute_percentage_error(testYa[0], testYp)
print('Actual test Score: %.2f SMAPE %.2f RMSE %.2f MAE' % (a_testSMAPE, a_testRMSE, a_testMAE))
testMAE = mean_absolute_error(testY[0], testPredict[:,0])
testRMSE = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
testSMAPE = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f SMAPE %.2f RMSE %.2f MAE' % (testSMAPE, testRMSE, testMAE))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
