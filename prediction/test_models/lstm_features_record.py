import math
from time import gmtime,strftime
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
from sqlalchemy.orm import sessionmaker
from sqlalchemy import *
from argparse import ArgumentParser
import configuration as cfg

# command line arguments
parser = ArgumentParser(description='Required arguments')
parser.add_argument('-p','--patient', help='Patient ID', required=True)
### svn revision for logging the results ###
parser.add_argument('-r', '--revision', help='Source code revision', required=False)
parser.add_argument('-d', '--debug', action='store_true')
#parser.add_argument('--feature', dest='feature', action='store_true')
args = parser.parse_args()

#print("args: {}".format(args))

##### Global vars \o/ #####
# patient id mappings
patients = {'8':'1','9':'5','10':'7','11':'3','12':'6','13':'4','14':'9','15':'2','16':'8','17':'10'}
# date of first glucose datapoint
min_date = None
max_date = None
# indices of not missing values
idx = []
# fix random seed for reproducibility
numpy.random.seed(7)

# static parameters
_datafolder="data/"

## timestamp representation ##
discritized = True

##### Parameters #####
look_back = 10
interpolate = True
split_ratio = 0.66
epochs = 50

DEBUG = args.debug
num_units = 4 # number of hidden units in RNN cell
if DEBUG:
  look_back = 8
  split_ratio = 0.9
  epochs = 2


### Model ####
models = {'lstm': LSTM, 'rnn': SimpleRNN}
modelName = 'rnn'

### Database connection ###
engine = create_engine("mysql+mysqldb://{user}:{passwd}@{host}/{db}".format(**cfg.data['database']), echo=DEBUG)

connection = engine.connect()
Session = sessionmaker(bind=connection)
session = Session()
# define meta information
metadata = MetaData(bind=engine)

experimentT = None


##### Functions #####

def createExperimentTable():
  global experimentT
  # create Experiment table if not existed
  # corresponding Prediction table will be automatically created
  if not engine.dialect.has_table(engine, "Experiment"):
    # Create a table with the appropriate Columns
    Table("Experiment", metadata,
          Column('id', Integer, primary_key=True, nullable=False),
          Column('model', String(20)), Column('parameters', String(300)),
          Column('timestamp', DateTime), Column('SMAPE', Float), Column('RMSE', Float), Column('svn_rev', Integer),
          Column('patientID', Integer))
    metadata.create_all()
    experimentT = Table('Experiment', metadata, autoload=True)
  else: experimentT = Table('Experiment', metadata, autoload=True)


def recordParameters():
  s = "\t"
  seq =("look_back: " + str(look_back), "interpolate: " + str(interpolate),
        "split_ratio: " + str(split_ratio), "epochs: " + str(epochs),
        "num_units: " + str(num_units))
  p = s.join(seq)
  print p
  return p

def read_csv(c, how='mean',name="",resample=False):
  global min_date
  global max_date
  global idx
  csv = pd.read_csv(c,delimiter=";")
  df = pd.DataFrame(csv)
  df['date'] = pd.to_datetime(df['date'])
  s = df.set_index('date')
  # IMPORTANT: sort by date
  s = s.sort_index()

  # select the time only from 2017-03-01
  # where we have stable measurement data
  s = s["2017-03-01":]
  if resample: s = s.resample('H', how)
  # compute min date once when loading glucose file
  if name == "bg":
    min_date = min(s.index)
    max_date = max(s.index)
    print min_date
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
    if row["Patient"] == int(patients[args.patient]):
      dict1 = {}
      dict2 = {"start" : row["dA"] + " " + row["tAstart"]}
      dict3 = {"AkCal" : row["AkCal"]}
      dict1.update(dict2)
      dict1.update(dict3)
      row_list.append(dict1)

  df = pd.DataFrame(row_list)
  df['start'] = pd.to_datetime(df['start'])
  s = df.set_index('start')
  # IMPORTANT: sort by date
  s = s.sort_index()

  # select the time only from 2017-03-01
  # where we have stable measurement data
  s = s["2017-03-01":]
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
    print "test values: {}".format(testValues)
    # if we deleted the NaNs, there is nothing to do here
    return numpy.array(testValues)
  testResults = []
  trainSize = len(trainY)
  testSize = len(testY)
  print("indices of actual values: {}".format(idx))
  print("trainY size: {}".format(trainSize))
  for i in idx:
    i = i - look_back - trainSize
    if i < 0:
      continue
    # we are missing one instance at the very end, since we need a label from the next value always
    if i > testSize - 1:
      continue
    testResults.append(testValues.item(i))
  print "test results: {}".format(testResults)
  return numpy.array(testResults)


# for experiment recording / tracing purpose
def get_svn_rev():
  return args.revision

##### CODE #####

# check if Experiment table is existed or else create it
createExperimentTable()

# load data
# blooglucose dataframe

bg_raw = read_csv(_datafolder+args.patient + ".bloodglucose.h",name="bg")
bg = read_csv(_datafolder+args.patient + ".bloodglucose.h",name="bg",resample=True)
# insulin dataframe
il = read_csv(_datafolder+args.patient + ".insulin.h",how="sum",resample=True)
# carbohydrate dataframe
ca = read_csv(_datafolder+args.patient + ".carbohydrate.h", how="sum",resample=True)
# akCal dataframe
ak = read_akcal(_datafolder + "aktivitaet.csv")

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
#model.add(Model.LSTM.value(4))
model.add(models[modelName](num_units, input_shape=(look_back,5)))
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

assert len(testYp) == len(testYa), "Test length equality of actual and predicted values"
print"Raw BG: {}".format(bg_raw.values)

print("Test values: {}".format(testYa))

print("Predicted values: {}".format(testYp))


#rmse = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print rmse
actualRMSE = math.sqrt(mean_squared_error(testYa, testYp))
print("RMSE: {}".format(actualRMSE))

actualSMAPE = mean_absolute_percentage_error(testYa, testYp)
print("SMAPE: {}".format(actualSMAPE))


# writing parameters and results to database

i = insert(experimentT)
i = i.values({"model" : modelName,
              "parameters" : recordParameters(),
              "timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              "SMAPE": actualSMAPE,
              "RMSE": actualRMSE,
              "svn_rev": get_svn_rev(),
              "patientID": args.patient
              })
res = session.execute(i)

experimentID = res.lastrowid

session.commit()
print ("Type of testYp {}".format(type(testYp)))
## write to prediction table ##
pdf = pd.DataFrame.from_records(testYp, columns=["value"])
pdf['experimentID'] = experimentID
# fix horizon as hour
pdf['horizon'] = 'H'
# prediction position list
# last position is always skipped
# the first prediction position can vary
# depends on the look_back
p_pos = range(len(bg_raw) - len(testYp) -1,len(bg_raw)-1)
print "bg raw: %f" %len(bg_raw)
print "test length: %f" %len(testYp)
pdf['pos'] = p_pos

pdf.to_sql(name='Prediction', con=engine, if_exists='append', index=False, index_label=None)


## close all sessions and connections ##
session.close_all()
engine.dispose()
