import pandas as pd
import sys
import numpy
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# a sole script for testing purpose
# input: patientID
# output average baseline

def mean_absolute_percentage_error(y_true, y_pred):
      return numpy.mean(numpy.abs(y_true - y_pred) / ((numpy.abs(y_true) + numpy.abs(y_pred))/2)) * 100

split_ratio = 0.66
csv = pd.read_csv("../data/"+ sys.argv[1] + ".bloodglucose.h",delimiter=";")
df = pd.DataFrame(csv)
df['date'] = pd.to_datetime(df['date'])
s = df.set_index('date')
#s = s["2017-03-01":]
min_date = min(s.index)
max_date = max(s.index)
print("number of data points: {}".format(len(s)))
s = s.resample('H', 'mean')
s = s[min_date:max_date]
print("number of data points after re- and subsampling: {}".format(len(s)))

X = s.values
size = int(len(X) * split_ratio)
train, test = X[0:size], X[size:len(X)]
print("number of test instances including nan: {}".format(len(test)))
# remove missing values
train = train[numpy.logical_not(numpy.isnan(train))]
test = test[numpy.logical_not(numpy.isnan(test))]
print("number of test instances: {}".format(len(test)))

avg =  numpy.mean(train)
testp = [avg] * len(test)

actualRMSE = math.sqrt(mean_squared_error(test, testp))
print("RMSE: {}".format(actualRMSE))

actualSMAPE = mean_absolute_percentage_error(test, testp)
print("SMAPE: {}".format(actualSMAPE))

