import MySQLdb
import sys
import pandas as pd
import math
from pandas.io import sql as psql
from sqlalchemy import create_engine
import configuration as cfg
import getopt
import datetime

# patien id mappings
patients = {'8':'1','9':'5','10':'7','11':'3','12':'6','13':'4','14':'9','15':'2','16':'8','17':'10'}
names = {"Carste", "Ralf", "Sally", "Neil", "Frieda", "Amos", "Layla", "Johannes", "Katrin", "Guiliana"}

engine = create_engine("mysql+mysqldb://{user}:{passwd}@{host}/{db}".format(**cfg.data['database']))

min_date = None
max_date = None


def create_db():
  dbname = cfg.data['database']['db']
  engine.execute("CREATE DATABASE "  + dbname)
  engine.execute("USE " + dbname)

def drop_db():
  dbname = cfg.data['database']['db']
  engine.execute("DROP DATABASE "  + dbname)
  
def import_patient():
  plist = pd.DataFrame(list(patients.iteritems()), columns=['id','ActivityID'])
  plist['id'] = plist['id'].astype(int)
  plist.set_index('id',inplace=True)
  plist['name'] = names
  csv = pd.read_csv("data/patient.meta",delimiter=";")
  metadf = pd.DataFrame(csv)
  metadf['id'] = metadf['id'].astype(int)
  metadf.set_index('id',inplace=True)

  plist = plist.merge(metadf, how='outer', left_index=True, right_index=True)  
  plist.to_sql(name='BG_Patient', con=engine, if_exists='append', index=True, index_label=None)
  print plist

def read_csv(c, how='mean',name="",resample=False):
  global min_date
  global max_date
  
  csv = pd.read_csv(c,delimiter=";")
  df = pd.DataFrame(csv)
  df['date'] = pd.to_datetime(df['date'])
  s = df.set_index('date')
  # IMPORTANT: sort by date
  s = s.sort_index()
  # TODO: don't limit to march and ongoing
  # temp fix
  #s = s["2017-03-01":]
  if resample: s = s.resample('H', how)
  if name == "bg":
    min_date = min(s.index)
    max_date = max(s.index)
  print s
  s = s[min_date:max_date]
  print("number of data points after re- and subsampling: {}".format(len(s)))
  return s


def read_akcal(c):
  csv = pd.read_csv(c)
  row_list = []
  for idx, row in csv.iterrows():
    if row["Patient"] == int(patients[sys.argv[1]]):
      dict1 = {}
      dateString = row['dA']
      timeString = row['tAend'] # if tAend = '00:00:00' then date component needs to be incremented by 1 day
      date = datetime.strptime(dateString + " " + timeString, '%Y-%m-%d %H:%M:%S')
      if timeString == '00:00:00':
         date = date + timedelta(days=1)
      dict2 = {"start" : date.strftime("%Y-%m-%d %H:%M:%S")}
      dict3 = {"AkCal" : row["AkCal"]}
      dict1.update(dict2)
      dict1.update(dict3)
      row_list.append(dict1)

  df = pd.DataFrame(row_list)
  df['start'] = pd.to_datetime(df['start'])
  s = df.set_index('start')
  # IMPORTANT: sort by date
  s = s.sort_index()

  s = s.resample('H', 'sum')
  s = s[min_date:max_date]

  #s = s.fillna(0)
  return s

def read_steps(c):
  csv = pd.read_csv(c)
  row_list = []
  for idx, row in csv.iterrows():
    if row["Patient"] == int(patients[sys.argv[1]]):
      dict1 = {}
      dict2 = {"start" : row["dA"] + " " + row["tAstart"]}
      dict3 = {"steps" : row["S"]}
      dict1.update(dict2)
      dict1.update(dict3)
      row_list.append(dict1)

  df = pd.DataFrame(row_list)
  df['start'] = pd.to_datetime(df['start'])
  s = df.set_index('start')
  #s = s.resample('H', 'sum')
  s = s[min_date:max_date]

  #s = s.fillna(0)
  return s


# import insulin data for all patients and metadata
def import_insulin():
  # import raw without interpolation
  il = pd.read_csv("data/insulin_all.csv",delimiter=";")
  il.reset_index(level=0, inplace=True)
  # column names are fixed from the txt file
  # "date";"patientID";"aeration";"value";"name";"type"
  il['patientID'] = il['patientID'].astype(int)
  il['date'] = pd.to_datetime(il['date'])
  # remark: composite index
  il.set_index(["date", "patientID"], inplace=True)
  #il['date'] = pd.to_datetime(il['date'])
  #print bg
  il.to_sql(name="BG_Insulin", con=engine, if_exists='replace', index=True, index_label=None)

# import carbohydate RAW and steps
def import_t(c, factor):
  # import raw without interpolation
  il = None
  if factor != "steps":
     il = read_csv("data/"+sys.argv[1] + "."+factor+".h",how="sum",resample=False)
  else: 
     il = read_steps("data/aktivitaet.csv")
  il['patientID'] = c
  il.reset_index(level=0, inplace=True)
  il.columns= ["date","value","patientID"]
  
  il['patientID'] = il['patientID'].astype(int)
  il['date'] = pd.to_datetime(il['date'])

  # remark: composite index
  il.set_index(["date", "patientID"], inplace=True)
  #il['date'] = pd.to_datetime(il['date'])
  # add prefix for server db
  factor = "BG_" + factor
  #print bg
  il.to_sql(name=factor, con=engine, if_exists='append', index=True, index_label=None)

def import_instance(c):
  # TODO: don't read in file twice
  # remark: this is for reading without interplolation
  bg = read_csv("data/"+sys.argv[1] + ".bloodglucose.h",name="bg")
  bg['patientID'] = c
  bg['pos'] = list(range(0,len(bg)))
  
  bg.reset_index(level=0, inplace=True) 
  bg.columns= ["date","gt-value","patientID","pos"]

  bg['patientID'] = bg['patientID'].astype(int)
  bg['date'] = pd.to_datetime(bg['date'])

  # remark: composite index
  bg.set_index(["date", "patientID"], inplace=True)
  #bg['date'] = pd.to_datetime(bg['date'])

  #print bg
  bg.to_sql(name='BG_Instance', con=engine, if_exists='append', index=True, index_label=None)

def import_timeseries(bg, c):
  dataset = bg.merge(il, how='outer', left_index=True, right_index=True)
  dataset = dataset.merge(ca, how='outer', left_index=True, right_index=True)
  dataset = dataset.merge(steps, how='outer', left_index=True, right_index=True)
  # remark: composite indices
  dataset.reset_index(level=0, inplace=True)
  dataset.columns= ["date","bloodglucose","insulin","carbohydrate","steps"]
  dataset['patientID'] = c
  dataset['patientID'] = dataset['patientID'].astype(int)
  dataset['date'] = pd.to_datetime(dataset['date'])  
  dataset.set_index(["date", "patientID"], inplace=True)
  # dont fill as 0 for database import
  #dataset = dataset.fillna(0)
 
  dataset.to_sql(name='BG_Timeseries', con=engine, if_exists='append', index=True, index_label=None)  

# load data
# TODO: only read these in when necessary
# blooglucose dataframe  
bg = read_csv("data/"+sys.argv[1] + ".bloodglucose.h",name="bg",resample=True)
# insulin dataframe
il = read_csv("data/"+sys.argv[1] + ".insulin.h",how="sum",resample=True)
# carbohydrate dataframe
ca = read_csv("data/"+sys.argv[1] + ".carbohydrate.h", how="sum",resample=True)
# akCal dataframe
# 18.07: remove as feature
# ak = read_akcal("data/aktivitaet.csv")
# number of steps dataframe
steps = read_steps("data/aktivitaet.csv")


# TODO: in case these share data, exectute them together. If faster execution is necessary for testing, implement testing mode properly (cut of data; read first k lines..)
if sys.argv[2] == "timeseries": import_timeseries(bg, sys.argv[1])
elif sys.argv[2] == "instance": import_instance(sys.argv[1])
elif sys.argv[2] == "patient" : import_patient()
elif sys.argv[2] == "carbohydrate" or sys.argv[2] == "steps": 
  import_t(sys.argv[1], sys.argv[2])
elif sys.argv[2] == "insulin": import_insulin()
#elif sys.argv[2] == "c": create_db()
#elif sys.argv[2] == "d": drop_db()
