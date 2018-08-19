# Rolling ARIMA
# on interplated data

import sys
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from numpy import ma
csv = pd.read_csv(sys.argv[1],delimiter=";")
df = pd.DataFrame(csv)
df['date'] = pd.to_datetime(df['date'])
s = df.set_index('date')
s = s["2017-03-01 00:00:00":"2017-03-31"]
min_date = min(s.index)
#print s
print("number of data points: {}".format(len(s)))
s = s.resample('H', 'mean')
s = s[min_date.replace(minute=0, second=0):"2017-03-31"]
print("number of data points after re- and subsampling: {}".format(len(s)))

if sys.argv[2] == "pad":
  print "performing padding interpolation"
  s_in = s.fillna(method="pad")
elif sys.argv[2]== "cubic":
  print "performing cubic spline interpolation"
  s_in = s.interpolate(method='cubic')
else:
  print "performing linear interpolation"
  s_in = s.interpolate(method='linear')
#idx for actual values
idx = []
cnt = 0
for index, row in s.iterrows():
  if not math.isnan(row[0]):
     idx.append(cnt)
     #print cnt, row[0]
  cnt += 1


X = s_in.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
print("number of test instances with nans: {}".format(len(test)))

history = [x for x in train]
predictions = list()
print idx
#actual value positions
testA = [] # actual test values
testidx = [] # positions 
for i in idx:
  if i <= len(train): continue
  if i - len(train) > len(test): continue
  # because seems some values are missing in the middle
  # safer to do from the end back
  rest = len(X) - i
  testA.append(test[len(test)-rest])
  testidx.append(len(test)-rest)
print testA
print("number of test instances without nans: {}".format(len(testA)))

predictA = [] # predictions on actual values
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
        if t in testidx:
           predictA.append(yhat)

for t in range(len(testA)):
  print('predicted=%f, expected=%f' % (testA[t], predictA[t]))

error = math.sqrt(mean_squared_error(test, predictions))
mse = math.sqrt(mean_squared_error(testA, predictA))
print('Test RMSE: %.3f' % error)
print('Test actual RMSE: %.3f' % mse)

def mean_absolute_percentage_error(y_true, y_pred):
	#y_true, y_pred = check_arrays(y_true, y_pred
	y_true = numpy.array(y_true)
	y_pred = numpy.array(y_pred)
	return numpy.mean(numpy.abs((y_true - y_pred) / (numpy.abs(y_true) + numpy.abs(y_pred)/2))) * 100

smape = mean_absolute_percentage_error(testA, predictA)
print('Test actual SMAPE: %.3f' % smape)

mae = mean_absolute_error(testA, predictA)
print('Test actual MAE: %.3f' % mae)


