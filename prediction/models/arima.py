import numpy
from statsmodels.tsa.arima_model import ARIMA as ARIMA_MODEL
import pandas as pd
from prediction.base_regressor import BaseRegressor
from util.measures import compute_performance_time_binned, compute_performance_meals


class ARIMA(BaseRegressor):
    ##### Parameters #####
    p = 5  # auto-regressive part of the model, incoporate past values
    d = 1  # integrated part, incoporate amount of differencing (i.e. the number of past time points to subtract from the current value)
    q = 0  # moving average,
    split_ratio = 0.66  # train / test split

    def __init__(self, patientId, dbConnection):
        super(ARIMA, self).__init__(patientId, dbConnection)

    def save_params(self):
        baseParams = self.save_base_params();
        seq = (baseParams, "p: " + str(self.p), "d: " + str(self.d), "q: " + str(self.q))
        return ";".join(seq)

    def predict(self):
        size = int(len(self.discretizedData) * self.split_ratio)
        train, test = self.discretizedData[0:size], self.discretizedData[size:len(self.discretizedData)]
        print("number of test instances including nan: {}".format(len(test)))
        # remove missing values
        train = train[numpy.logical_not(numpy.isnan(train))]
        test = test[numpy.logical_not(numpy.isnan(test))]
        print("number of test instances: {}".format(len(test)))

        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA_MODEL(history, order=(self.p, self.d, self.q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            size = int(len(self.glucose_data) * self.split_ratio)
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))

        # load timestamps
        ts_df = self.load_timestamps(self.con, self.patient_id)
        test_ts = ts_df.tail(len(test))
        results = dict()
        results['groundtruth'] = test
        timestamps = test_ts.index
        results['times'] = timestamps
        results['indices'] = test_ts['pos'].values
        results['predictions'] = predictions
        results['performance'] = compute_performance_time_binned(
            timestamps=timestamps,
            groundtruth=test,
            predictions=predictions)
        results['performance'].update(compute_performance_meals(
            timestamps=timestamps,
            groundtruth=test,
            predictions=predictions,
            carbdata=self.carb_data
        ))
        results['params'] = self.save_params()

        return results
