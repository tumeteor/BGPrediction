from prediction.base_regressor import BaseRegressor

from sklearn.metrics import mean_absolute_error, median_absolute_error
from util.measures import mean_squared_error
from sklearn import ensemble
from sklearn.model_selection import LeavePOut, KFold
import pandas
import numpy

class RandomForest(BaseRegressor):

    best_params = None

    param_grid = {"n_estimators": [100, 250, 300, 400, 500, 700, 1000],
                  "criterion": ["mae", "mse"],
                  "max_features": ["auto", "sqrt", "log2"],
                  "min_samples_leaf": range(1,6)}

    ### Model ####
    # Extratree seems to deal better for small dataset + overfitting
    models = {'rf': ensemble.RandomForestRegressor, 'et': ensemble.ExtraTreesRegressor}

    n_estimator = 300
    criterion = "mse"
    min_samples_leaf = 3 # small for ExtraTree is helpful


    def __init__(self, patientId, dbConnection, modelName):
        super(RandomForest, self).__init__(patientId, dbConnection)
        self.modelName = modelName
        #self.tune = True
        if self.modelName == "rf":
            self.min_samples_leaf = 4 # default parameter for Ranfom Forest
            self.n_estimator = 500

    def save_params(self):
        baseParams = self.save_base_params();
        params = ";".join(("n_estimator: " + str(self.n_estimator), "criterion: " + str(self.criterion), "min_samples_leaf: " + str(self.min_samples_leaf)))
        if self.tune: params = ";".join((baseParams, params, str(self.param_grid), "Best params: " + str(self.best_params)))
        return params


    def predict(self):
        if not self._customizeFeatureSet:
            # generate features
            data, y = self.extract_features()
        else:
            data, y, _featureDesp = self.extract_features(customizeFeatureSet=True)
        return self.predict_with_data(data, y)

    def incremental_confidence_eval(self, data, y):

        print len(data)

        for i in range(10, len(data)):
            if i % 5 != 0: continue
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)


    def train_with_night_filtering(self, data, y):
        import pandas as pd
        import datetime

        ts = numpy.array([item['time'] for item in self.glucose_data[1:]])
        assert (len(ts) == len(data))

        rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                         min_samples_leaf=self.min_samples_leaf)
        kf = KFold(n_splits=5)

        maes = {}
        rmses = {}
        no_preds_removed = {}
        actual_maes = []
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ts_train, ts_test = ts[train_index], ts[test_index]
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            actual_maes.append(median_absolute_error(y_test, predictions))

            for threshold_time in [3,4,5,6,7,8,9]:
                y_test_filtered = []
                predictions_filtered = []
                pred_removed = 0
                for j in range(0, len(y_test)):
                    # filter by night time
                    time = pd.Timestamp(ts_test[j]).tz_localize("UTC")
                    # TODO: check UTC+1
                    if time.time() < datetime.time(threshold_time):
                        # print time.hour
                        pred_removed += 1
                        continue
                    y_test_filtered.append(y_test[j])
                    predictions_filtered.append(predictions[j])
                if len(y_test_filtered) == 0: continue
                mae = median_absolute_error(y_test_filtered, predictions_filtered)
                rmse = mean_squared_error(y_test_filtered, predictions_filtered)

                if threshold_time in maes.keys(): maes[threshold_time].append(mae)
                else: maes[threshold_time] = [mae]
                if threshold_time in rmses.keys():
                    rmses[threshold_time].append(rmse)
                else:
                    rmses[threshold_time] = [rmse]
                if threshold_time in no_preds_removed.keys():
                    no_preds_removed[threshold_time].append(pred_removed)
                else:
                    no_preds_removed[threshold_time] = [pred_removed]
        import collections
        od_maes = collections.OrderedDict(sorted(maes.items()))

        for threshold_time, mae in od_maes.iteritems():
            print threshold_time, numpy.mean(mae), numpy.std(mae), numpy.mean(rmses[threshold_time]), numpy.std(rmses[threshold_time]), numpy.mean(actual_maes), numpy.mean(no_preds_removed[threshold_time])


    def train_with_confidence_filtering(self, data, y):

        rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                         min_samples_leaf=self.min_samples_leaf)
        kf = KFold(n_splits=5)

        maes = {}
        rmses = {}
        no_preds = {}
        actual_maes = []
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rf.fit(X_train, y_train)

            V_IJ, V_IJ_unbiased = self.confidence_cal(X_train, X_test, rf)
            predictions = rf.predict(X_test)

            assert (len(V_IJ_unbiased) == len(X_test))
            actual_maes.append(median_absolute_error(y_test, predictions))
            for threshold_variance in numpy.arange(0.0, 4.0, 0.2):
                y_test_filtered = []
                predictions_filtered = []
                for j in range(0, len(y_test)):
                    # filter by variance confidence
                    if V_IJ_unbiased[j] > threshold_variance: continue
                    y_test_filtered.append(y_test[j])
                    predictions_filtered.append(predictions[j])
                if len(y_test_filtered) == 0: continue
                mae = median_absolute_error(y_test_filtered, predictions_filtered)
                rmse = mean_squared_error(y_test_filtered, predictions_filtered)

                if threshold_variance in maes.keys(): maes[threshold_variance].append(mae)
                else: maes[threshold_variance] = [mae]
                if threshold_variance in rmses.keys():
                    rmses[threshold_variance].append(rmse)
                else:
                    rmses[threshold_variance] = [rmse]
                if threshold_variance in no_preds.keys():
                    no_preds[threshold_variance].append(len(predictions_filtered))
                else:
                    no_preds[threshold_variance] = [len(predictions_filtered)]
        import collections
        od_maes = collections.OrderedDict(sorted(maes.items()))

        for threshold_variance, mae in od_maes.iteritems():
            print threshold_variance, numpy.mean(mae), numpy.std(mae), numpy.mean(rmses[threshold_variance]), numpy.std(rmses[threshold_variance]), numpy.mean(actual_maes), numpy.mean(no_preds[threshold_variance])



        # column_labels = ['run1', 'run2', 'run3', 'run4', 'run5']
        # df = pandas.DataFrame(vars, columns=column_labels)
        # print df.to_csv(index=False)

    def incremental_train_for_std(self, data, y):
        import pandas as pd
        import datetime

        timestamps = [item['time'] for item in self.glucose_data[1:]]
        assert (len(timestamps) == len(data))

        for i in range (10, len(data)):
            if i % 5 != 0: continue
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)
            kf = KFold(n_splits=5)
            sub_data = data[0:i]
            sub_y = y[0:i]
            sub_ts = numpy.array(timestamps[0:i])

            maes = []
            rmses = []
            for train_index, test_index in kf.split(sub_data):
                X_train, X_test = sub_data[train_index], sub_data[test_index]
                y_train, y_test = sub_y[train_index], sub_y[test_index]
                ts_train, ts_test = sub_ts[train_index], sub_ts[test_index]
                rf.fit(X_train, y_train)

                V_IJ, V_IJ_unbiased = self.confidence_cal(X_train, X_test, rf)
                predictions = rf.predict(X_test)
                #print "variance: {}".format(V_IJ_unbiased)

                assert (len(V_IJ_unbiased) == len (X_test))

                mae = median_absolute_error(y_test,predictions)
                rmse = mean_squared_error(y_test, predictions)

                # mae = median_absolute_error(y_test, predictions)

                maes.append(mae)
                rmses.append(rmse)


            std = numpy.std(maes)

            avg_mae = numpy.mean(maes)

            print i, avg_mae, std, numpy.mean(rmses), numpy.std(rmses)

    def incremental_train(self, data, y):
        import pandas as pd
        import datetime

        timestamps = [item['time'] for item in self.glucose_data[1:]]
        assert (len(timestamps) == len(data))

        for i in range (10, len(data)):
            if i % 5 != 0: continue
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)
            kf = KFold(n_splits=5)
            sub_data = data[0:i]
            sub_y = y[0:i]
            sub_ts = numpy.array(timestamps[0:i])

            maes = []
            for train_index, test_index in kf.split(sub_data):
                X_train, X_test = sub_data[train_index], sub_data[test_index]
                y_train, y_test = sub_y[train_index], sub_y[test_index]
                ts_train, ts_test = sub_ts[train_index], sub_ts[test_index]
                rf.fit(X_train, y_train)

                V_IJ, V_IJ_unbiased = self.confidence_cal(X_train, X_test, rf)
                predictions = rf.predict(X_test)
                #print "variance: {}".format(V_IJ_unbiased)

                assert (len(V_IJ_unbiased) == len (X_test))
                threshold_variance = 1
                y_test_filtered = []
                predictions_filtered = []

                for j in range(0, len(y_test)):
                     # filter by night time
                     time = pd.Timestamp(ts_test[j]).tz_localize("UTC")
                     # TODO: check UTC+1
                     if time.time() < datetime.time(5):
                         #print time.hour
                         continue

                     # filter by variance confidence
                     if V_IJ[j] > threshold_variance: continue
                     y_test_filtered.append(y_test[j])
                     predictions_filtered.append(predictions[j])
                if len(y_test_filtered) == 0: continue
                mae = median_absolute_error(y_test_filtered,predictions_filtered)

                # mae = median_absolute_error(y_test, predictions)

                maes.append(mae)


            std = numpy.std(maes)

            avg_mae = numpy.mean(maes)

            print i, avg_mae, std

    def incremental_train_with_temporal_order(self, data, y):

        print len(data)

        for i in range(10, len(data)):
            if i % 5 != 0: continue
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)
            sub_data = data[0:i]
            sub_y = y[0:i]

            train_data = sub_data[0:i-1]
            train_y = sub_y[0:i-1]
            test_data = sub_data[i-1:]
            test_y = sub_y[i-1:]

            rf.fit(train_data,train_y)
            V_IJ_unbiased = self.confidence_cal(train_data, test_data, rf)
            print "variance: {}".format(V_IJ_unbiased)
            predictions = rf.predict(test_data)

            mae = mean_absolute_error(test_y, predictions)

            print i, mae

    def incremental_lookback(self, data, y):

        test_data = data[len(data)-4:]
        test_y = y[len(data)-4:]

        train_data = data[0:len(data)-5]
        train_y = y[0:len(data)-5]


        for i in range(10, len(train_data)):
            if i % 5 != 0: continue
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)
            sub_data = train_data[len(train_data)-i:]
            sub_y = train_y[len(train_y)-i:]

            rf.fit(sub_data, sub_y)
            predictions = rf.predict(test_data)

            mae = median_absolute_error(test_y, predictions)

            #V_IJ_unbiased = self.confidence_cal(train_data,test_data,rf)

            print i, mae

    def confidence_cal(self, train_data, test_data, rf):
        import forestci as fci
        # calculate inbag and unbiased variance
        inbag = fci.calc_inbag(train_data.shape[0], rf)
        V_IJ, V_IJ_unbiased = fci.random_forest_error(rf, train_data, test_data)

        return V_IJ, V_IJ_unbiased


    def predict_with_data(self, data, y, _featureDesp="all"):

        assert (len(data) == len(y))

        if self.patient_id <= 11: return

        # confident intervals with small data
        self.train_with_night_filtering(data, y)


        print "Results for Patient {}".format(self.patient_id)



