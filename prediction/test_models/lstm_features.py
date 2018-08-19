import math
import numpy
from pandas import read_csv
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Masking
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import ma
#from pykalman import KalmanFilter

##### Global vars \o/ #####
# patien id mappings
patients = {'8':'1','9':'5','10':'7','11':'3','12':'6','13':'4','14':'9','15':'2','16':'8','17':'10'}
# date of first glucose datapoint
min_date = None
max_date = None
# indices of not missing values
idx = []
# fix random seed for reproducibility
numpy.random.seed(7)

##### Parameters #####
look_back = 4
interpolate = False
split_ratio = 0.66
epochs = 50
DEBUG = True
#DEBUG = False
if DEBUG:
  look_back = 2
  split_ratio = 0.9
  epochs = 2

##### Functions #####
def read_csv(c, how='mean',name=""):
  global min_date
  global max_date
  global idx
  csv = pd.read_csv(c,delimiter=";")
  df = pd.DataFrame(csv)
  df['date'] = pd.to_datetime(df['date'])
  s = df.set_index('date')
  s = s.resample('H', how)
  # compute min date once when loading glucose file
  if name == "bg":
    min_date = min(s.index).replace(minute=0, second=0)
    max_date = max(s.index).replace(minute=0, second=0)
  s = s[min_date:max_date]
  print("number of data points after re- and subsampling: {}".format(len(s)))
  if name == "bg":
    #idx for actual values
    cnt = 0
    for index, row in s.iterrows():
      if not math.isnan(row[0]):
        if row[0] > 0.0:
          idx.append(cnt)
      cnt += 1
    # pad missing values
  return s

def read_akcal(c):
  csv = pd.read_csv(c)
  row_list = []
  for idx, row in csv.iterrows():
    if row["Patient"] == int(patients[sys.argv[1]]):
      dict1 = {}
      dict2 = {"start" : row["dA"] + " " + row["tAstart"]}
      dict3 = {"AkCal" : row["AkCal"]}
      dict1.update(dict2)
      dict1.update(dict3)
      row_list.append(dict1)

  df = pd.DataFrame(row_list)
  df['start'] = pd.to_datetime(df['start'])
  s = df.set_index('start')
  s = s.resample('H', 'sum')
  s = s[min_date:max_date]
  s = s.fillna(0)
  return s

def timeofday(c):
  # sleep: 12-5, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
  times_of_day = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5 ]
  return times_of_day[c.hour-1]

def createPeriods():
  times = []
  for idx, row in bg.iterrows():
    times.append(timeofday(idx))
  return times

def build_dataset():
  global bg
  if interpolate:
    bg = bg.fillna(method="pad")
  times = createPeriods()
  bg['period'] = times
  dataset = bg.merge(il, how='outer', left_index=True, right_index=True)
  dataset = dataset.merge(ca, how='outer', left_index=True, right_index=True)
  dataset = dataset.merge(ak, how='outer', left_index=True, right_index=True)
  dataset = dataset.fillna(0)
  print("build_dataset(): Joined time series data. Number of rows: {}".format(len(bg)))
  return create_dataset(dataset.values, look_back)

# Add time dimension to the dataset by transforming it from 2D to 3D
def create_dataset(dataset, look_back=1):
  dataX, dataY = [] * dataset.shape[1], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back)]
    #a = a.ravel()
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  dataX = numpy.array(dataX)
  dataY = numpy.array(dataY)
  if not interpolate:
    # remove the missing values (glucose == 0)
    idxTrueValues = numpy.nonzero(numpy.array(dataY))[0]
    print idxTrueValues
    print type(idxTrueValues)
    dataY = dataY[idxTrueValues]
    dataX = dataX[idxTrueValues]
  print("create_dataset(): created dataset with lookback {}. Instances: {} ".format(look_back,len(dataX)))
  return dataX, dataY

def mean_absolute_percentage_error(y_true, y_pred):
  return numpy.mean(numpy.abs(y_true - y_pred) / ((numpy.abs(y_true) + numpy.abs(y_pred))/2)) * 100


def getTestValues(testValues):
  """
  Transform indices of original ground truth glucose values and get the corresponding values from predictions and test sets
  """
  if not interpolate:
    # if we deleted the NaNs, there is nothing to do here
    return numpy.array(testValues)
  testResults = []
  trainSize = len(trainY)
  testSize = len(testY)
  for i in idx:
    i = i - look_back - trainSize
    if i < 0:
      continue
    # we are missing one instance at the very end, since we need a label from the next value always
    if i > testSize - 1:
      continue
    testResults.append(testValues.item(i))
  return numpy.array(testResults)

##### CODE #####

# load data
# blooglucose dataframe
bg = read_csv("../data/"+sys.argv[1] + ".bloodglucose.h",name="bg")
# insulin dataframe
il = read_csv("../data/"+sys.argv[1] + ".insulin.h","sum")
# carbohydrate dataframe
ca = read_csv("../data/"+sys.argv[1] + ".carbohydrate.h", "sum")
# akCal dataframe
ak = read_akcal("../data/aktivitaet.csv")

# join time series and add time dimension
X,Y = build_dataset()
#X = X.astype('float32')

# TODO: FIX! transform dataset before split; DO NOT IGNORE TEST DATA EVER
#trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)

# TODO: fix split when not interpolating
train_size = int(len(X) * split_ratio)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]
# reshape into X=t and Y=t+1

# TODO: remove the missing values here; unless we want to interpolate: then do it before transforming the data

# scale all the data \o/
xScaler = MinMaxScaler(feature_range=(0,1))
yScaler = MinMaxScaler(feature_range=(0,1))

# reshape to 2D  for MinMaxScaler
train_nsamples, train_nx, train_ny = trainX.shape
d2_trainX = trainX.reshape((train_nsamples,train_nx*train_ny))

test_nsamples, test_nx, test_ny = testX.shape
d2_testX = testX.reshape((test_nsamples,test_nx*test_ny))

trainX = xScaler.fit_transform(d2_trainX)
testX = xScaler.transform(d2_testX)

trainY = yScaler.fit_transform(trainY)
testY = yScaler.fit_transform(testY)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (train_nsamples, train_nx, train_ny))
testX = numpy.reshape(testX, (test_nsamples, test_nx, test_ny))

#actual values
testYa = getTestValues(testY)

# create and fit the LSTM network
model = Sequential()
# input shape: (look_back, nfeatures)
model.add(Masking(mask_value=0., input_shape=(look_back, 5)))
model.add(LSTM(4))
#model.add(SimpleRNN(4, input_shape=(look_back,5)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
# make predictions
#trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print("#Test data points (w/ padding): {}; #instances: {}; #instances: {};".format(len(testY), len(X), len(trainX)))

testYp = getTestValues(testPredict)

testYp = yScaler.inverse_transform(testYp)
testYa = yScaler.inverse_transform(testYa)

print("Test values: {}".format(testYa))
print("Predicted values: {}".format(testYp))
#rmse = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print rmse
actualRMSE = math.sqrt(mean_squared_error(testYa, testYp))
print("RMSE: {}".format(actualRMSE))

actualSMAPE = mean_absolute_percentage_error(testYa, testYp)
print("SMAPE: {}".format(actualSMAPE))
