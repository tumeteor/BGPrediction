import MySQLdb
import sys
import pandas as pd
import math
from pandas.io import sql as psql
from sqlalchemy import create_engine
import config as cfg
import getopt

#engine = create_engine("mysql+mysqldb://{user}:{passwd}@{host}/{db}".format(**cfg.data['database']))
engine = create_engine("mysql+mysqldb://root:root@localhost/BGPrediction")
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

  #s = s.fillna(0)
  return s

def read_csv(c, how='mean',name="",resample=False):
  global min_date
  global max_date

  csv = pd.read_csv(c,delimiter=";")
  df = pd.DataFrame(csv)
  df['date'] = pd.to_datetime(df['date'])
  s = df.set_index('date')
  # FIXED: don't limit to march and ongoing
  s = s["2017-03-01":]
  if resample: s = s.resample('H', how)
  if name == "bg":
    min_date = min(s.index)
    max_date = max(s.index)
  print s
  s = s[min_date:max_date]

  print("number of data points after re- and subsampling: {}".format(len(s)))
  return s


def importInstance():
# Quick fix: don't read in file twice: this is for read with no resample
  bg = read_csv("data/" + sys.argv[1] + ".bloodglucose.t",name="bg")
  bg['pos'] = list(range(0,len(bg)))
  bg.rename(columns={"date":"time","bloodglucose":"gt-value","patientID":"patientID"})
  #print bg
  bg.to_sql(name='BG_Instance2', con=engine, if_exists='append', index=True, index_label=None)

importInstance()

