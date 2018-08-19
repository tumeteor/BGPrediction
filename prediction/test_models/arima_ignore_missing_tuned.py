# Rolling ARIMA

import sys
import numpy
import warnings
from time import gmtime,strftime
import matplotlib.pyplot as plt
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from argparse import ArgumentParser
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData, create_engine, select, insert, update, table
import pysvn

modelName = "ARIMA"
parser = ArgumentParser(description='Required arguments')

parser.add_argument('-p','--patient', help='Patient ID', required=True)
### svn credential ###
# needed when encounter with the call_back login exception
parser.add_argument('-u','--user', help='Your svn username', required=True)
parser.add_argument('-w','--pwd', help='Your svn password', required=True)
args = parser.parse_args()

## timestamp representation ##
discritized = True


### Database connection ###
engine = create_engine("mysql+mysqldb://root:root@localhost/BloodGlucosePrediction")

connection = engine.connect()
Session = sessionmaker(bind=connection)
session = Session()
# define meta information
metadata = MetaData(bind=engine)

experimentT = Table('Experiment', metadata, autoload=True)


##### Parameters #####
# p auto-regressive part of the model, incoporate past values
# d integrated part, incoporate amount of differencing (i.e. the number of past time points to subtract from the current value) 
# q moving average, 

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 1)
q_values = range(0, 1)
warnings.filterwarnings("ignore")

split_ratio = 0.66 # train / test split
def recordParameters(best_cfg):
  return ','.join(best_cfg) + "\t" + "split_ratio: " + str(split_ratio)

# for experiment recording / tracing purpose
def get_svn_rev():
  client = pysvn.Client()
  client.callback_get_login = getLogin
  rev = client.update('.')
  return str(rev[0]).split(" ")[-1][:-1]

# to get over the callback_get_login exception thingy
def getLogin(realm, username, may_save):
    return True, args.user, args.pwd, False


csv = pd.read_csv("../data/"+args.patient + ".bloodglucose.h",delimiter=";")
df = pd.DataFrame(csv)
df['date'] = pd.to_datetime(df['date'])
s = df.set_index('date')
s = s["2017-03-01":]
min_date = min(s.index)
max_date = max(s.index)
print("number of data points: {}".format(len(s)))
s = s.resample('H', 'mean')
s = s[min_date:max_date]
print("number of data points after re- and subsampling: {}".format(len(s)))

X = s.values
def mean_absolute_percentage_error(y_true, y_pred):
        #y_true, y_pred = check_arrays(y_true, y_pred
        y_true = numpy.array(y_true)
        y_pred = numpy.array(y_pred)
        return numpy.mean(numpy.abs((y_true - y_pred) / (numpy.abs(y_true) + numpy.abs(y_pred)/2))) * 100

# evaluate ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X,order):
  size = int(len(X) * split_ratio)
  train, test = X[0:size], X[size:len(X)]
  print("number of test instances including nan: {}".format(len(test)))
  # remove missing values
  train = train[numpy.logical_not(numpy.isnan(train))]
  test = test[numpy.logical_not(numpy.isnan(test))]
  print("number of test instances: {}".format(len(test)))

  history = [x for x in train]
  predictions = list()
  for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
  smape = mean_absolute_percentage_error(test, predictions)
  return smape, test, predictions

# grid search on parameters
def evaluate_models(dataset, p_values, d_values, q_values):
   dataset = dataset.astype('float32')
   best_score, best_cfg, best_test, best_predictions = float("inf"), None, None, None
   for p in p_values:
     for d in d_values:
       for q in q_values:
	   order = (p,d,q)
	   smape, test, predictions = evaluate_arima_model(dataset, order)
	   if smape < best_score:
	     best_score, best_cfg, best_test, best_predictions = smape, order, test, predictions
	     print('ARIMA%s MSE=%.3f' % (order,smape))

   print('Best ARIMA%s SMAPE = %.3f' % (best_cfg, best_score))
   return best_cfg, best_score, best_test, best_predictions

best_cfg, smape, test, predictions = evaluate_models(X, p_values, d_values, q_values)
print best_cfq
rmse = math.sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)

smape = mean_absolute_percentage_error(test, predictions)
print('SMAPE: %.3f' % smape)

# writing parameters and results to database

i = insert(experimentT)
i = i.values({"model" : modelName,
              "parameters" : recordParameters(best_cfg),
              "timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              "SMAPE": smape,
              "RMSE": rmse,
              "svn_rev": get_svn_rev(),
              "patientID": args.patient
              })

res = session.execute(i)

experimentID = res.lastrowid

session.commit()
## close all sessions and connections ##
session.close_all()
engine.dispose()

