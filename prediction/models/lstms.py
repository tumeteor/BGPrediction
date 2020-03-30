from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Masking
from sklearn.preprocessing import MinMaxScaler
import logging
import pandas as pd
import numpy
from util.measures import compute_performance_time_binned
from util.measures import compute_performance_meals
from util.measures import getTimeBinInt
from prediction.base_regressor import BaseRegressor
import math
class LSTMs(BaseRegressor):

    ### Model ####
    models = {'lstm': LSTM, 'rnn': SimpleRNN}
    # multivariate time series
    dataset = None

    ##### Parameters #####
    interpolated = False
    epochs = 10

    DEBUG = False
    addTimeofDay = True
    no_features = 4 # number of features for input shape
    num_units = 20 # number of hidden units in RNN cell
    if DEBUG:
        look_back = 2
        split_ratio = 0.9
        epochs = 2

    discretized = True
    if not discretized:
        # regression
        look_back = 0


    # fix random seed for reproducibility
    # IMPORTANT: https://github.com/fchollet/keras/issues/439
    numpy.random.seed(7)

    def __init__(self, patientId, dbConnection,modelName):
        super(LSTMs, self).__init__(patientId, dbConnection)
        self.modelName = modelName

        # indices of not missing values
        self.idx = []



    def save_params(self):
        baseParams = self.save_base_params();
        seq = (baseParams, "discretized: " + str(self.discretized), "interpolated: " + str(self.interpolated), "epochs: " + str(self.epochs), "addTimeofDay: " + str(self.addTimeofDay), "no_features: " + str(self.no_features),
               "num_units: " + str(self.num_units), "debug: " + str(self.DEBUG))
        params = ";".join(seq)
        return params


    # Add time dimension to the dataset by transforming it from 2D to 3D
    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [] * dataset.shape[1], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back)]
            #a = a.ravel()
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        dataX = numpy.array(dataX)
        dataY = numpy.array(dataY)
        if not self.interpolated:
            # remove the missing values (glucose == 0)
            idxTrueValues = numpy.nonzero(numpy.array(dataY))[0]
            dataY = dataY[idxTrueValues]
            dataX = dataX[idxTrueValues]
        print("create_dataset(): created dataset with lookback {}. Instances: {} ".format(look_back,len(dataX)))
        return dataX, dataY


    @staticmethod
    def create_periods(df):
        times = []
        for idx, row in df.iterrows():
            times.append(getTimeBinInt(idx))
        return times


    def get_test_values(self, testValues, trainSize, testSize):
        print "testValue: {}, trainSize: {}, testSize: {}".format(len(testValues), trainSize, testSize)
        """
        Transform indices of original ground truth glucose values and get the
        corresponding values from predictions and test sets
        """
        if not self.interpolated:
            # if we deleted the NaNs, there is nothing to do here
            return numpy.array(testValues)
        testResults = []
        for i in self.idx:
            j = i - self.look_back - trainSize
            if j < 0:
                continue
            # we are missing one instance at the very end, since we need a label from the next value always
            if j > testSize - 1:
                continue
            testResults.append(testValues.item(j))

        print ("length of testResults: {}".format(len(testResults)))
        return numpy.array(testResults)


    # load multivariate time series data
    # LSTM shows a good performance on patients with denser measurements e.g., 13. Hence,
    # TODO: integrate with new features

    def load_time_series(self, con, patientId):
        print("Loading time series  data for patient {}".format(patientId))
        df = None
        with con:
            cur = con.cursor()
            query = "SELECT date, bloodglucose as 'bg', insulin as 'il', carbohydrate as 'ca', Steps FROM BG_Timeseries " \
                    "WHERE patientID = {patientId} and date > '2017-02-25' ".format(patientId=patientId)
            df = pd.read_sql(query,con)


        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        # IMPORTANT: sort by date
        df = df.sort_index()
        periods = self.create_periods(df)
        if self.addTimeofDay: df['period'] = periods
        # select the time only from 2017-03-01
        # where we have stable measurement data
        df = df["2017-02-25":]

        min_date = min(df['bg'].to_frame().dropna().index)
        max_date = max(df['bg'].to_frame().dropna().index)
        print "min date: {}".format(min_date)
        print "max date: {}".format(max_date)
        df = df[min_date:max_date]
        # idx for actual values
        cnt = 0
        for index, row in df.iterrows():
            if not math.isnan(row[0]):
                if row[0] > 0.0:
                    self.idx.append(cnt)
            cnt += 1
        # pad missing values
        if self.interpolated: df['bg'] = df['bg'].fillna(method="pad")
        '''
        skip all nan values from BG
        '''
        df = df[numpy.isfinite(df['bg'])]
        df = df.fillna(0)
        print df

        return self.create_dataset(df.values, self.look_back)

    def load_continuous_data(self):
        features, y = self.extract_features()
        y = y[:,None]
        print y.shape
        print features.shape
        concatData = numpy.concatenate((y, features),
                                       axis=1)
        self.no_features = features.shape[1]
        self.look_back = 1
        return self.create_dataset(concatData, self.look_back)


    def predict(self):
        """
        load in data and model_name, whether LSTM or RNN
        :return: gt-values and predictions
        """
        X, Y = self.load_time_series(self.con, self.patient_id) if self.discretized else self.load_continuous_data()
        print "modelname %s" %self.modelName
        # TODO: fix split when not interpolating
        train_size = int(len(X) * self.split_ratio)
        test_size = len(X) - train_size
        trainX, testX = X[0:train_size], X[train_size:len(X)]
        trainY, testY = Y[0:train_size], Y[train_size:len(Y)]
        train_size = len(trainY)
        test_size = len(testY)

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

        trainY = trainY.reshape(-1,1)
        testY = testY.reshape(-1,1)

        trainY = yScaler.fit_transform(trainY)
        testY = yScaler.fit_transform(testY)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (train_nsamples, train_nx, train_ny))
        testX = numpy.reshape(testX, (test_nsamples, test_nx, test_ny))

        #actual values
        testYa = self.get_test_values(testY, train_size, test_size)

        # create and fit the LSTM network
        model = Sequential()

        #FIXED: we skip here the time period features
        if self.addTimeofDay: self.no_features += 1
        # input shape: (look_back, nfeatures)
        #model.add(Masking(mask_value=0., input_shape=(self.look_back, self.no_features)))
        #model.add(Model.LSTM.value(4))
        model.add(self.models[self.modelName](self.num_units, input_shape=(self.look_back,self.no_features)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=self.epochs, batch_size=1, verbose=2)
        # make predictions
        testPredict = model.predict(testX)

        print("#Test data points (w/ padding): {}; #instances: {}; # training instances: {};".format(len(testY), len(X), len(trainX)))
        # trace back predictions at actual values
        testYp = self.get_test_values(testPredict, train_size, test_size)


        testYp = yScaler.inverse_transform(testYp)
        testYa = yScaler.inverse_transform(testYa)
        print("Test values: {}".format(testYa))
        print("predicted values: {}".format(testYp))

        # load timestamps
        ts_df = self.load_timestamps(self.con, self.patient_id)
        # if self.interpolated:
        #     # skip last timestamp
        #     ts_df = ts_df[:-1]

        test_ts = ts_df.tail(len(testYa))
        # test_ts = test_ts.reset_index()
        results = dict()
        results['groundtruth'] = testYa
        timestamps = test_ts.index
        results['times'] = timestamps
        results['indices'] = test_ts['pos'].values
        results['predictions'] = testYp
        results['performance'] = compute_performance_time_binned(
            timestamps=timestamps,
            groundtruth=testYa,
            predictions=testYp)
        results['performance'].update(compute_performance_meals(
            timestamps=timestamps,
            groundtruth=testYa,
            predictions=testYp,
            carbdata=self.carb_data
        ))
        results['params'] = self.save_params()
        assert (len(testYa) == len(testYp))
        assert(len(timestamps) == len(testYp))
        return results
      

        

     
