from prediction.base_regressor import BaseRegressor
from util.measures import compute_performance_time_binned
from util.measures import compute_performance_meals, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, RFECV
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class RandomForest(BaseRegressor):
    best_params = None

    param_grid = {"n_estimators": [100, 250, 300, 400, 500, 700, 1000],
                  "criterion": ["mae", "mse"],
                  "max_features": ["auto", "sqrt", "log2"],
                  "min_samples_leaf": range(1, 6)}

    ### Model ####
    # Extratree seems to deal better for small dataset + overfitting
    models = {'rf': ensemble.RandomForestRegressor, 'et': ensemble.ExtraTreesRegressor}

    n_estimator = 300
    criterion = "mse"
    min_samples_leaf = 2  # small for ExtraTree is helpful

    def __init__(self, patientId, dbConnection, modelName):
        super(RandomForest, self).__init__(patientId, dbConnection)
        self.modelName = modelName
        # self.tune = True
        if self.modelName == "rf":
            self.min_samples_leaf = 4  # default parameter for Ranfom Forest
            self.n_estimator = 500

    def save_params(self):
        baseParams = self.save_base_params();
        params = ";".join(("n_estimator: " + str(self.n_estimator), "criterion: " + str(self.criterion),
                           "min_samples_leaf: " + str(self.min_samples_leaf)))
        if self.tune: params = ";".join(
            (baseParams, params, str(self.param_grid), "Best params: " + str(self.best_params)))
        return params

    def batchPredict(self):
        '''
        batch experiment for different feature subsets
        :return:
        '''
        X, y = self.Features.build_feature_matrix(self.look_back)
        _allSubsets = self.Features.feature_group_all_subsets(X)
        batch_results = []
        for subset, value in _allSubsets.iteritems():
            data = value[0]
            featureDesp = value[1]
            self.log.info("Feature Desp: {}".format(featureDesp))
            results = self.predictWithData(data, y, featureDesp)
            self._allFeatureDesp.append(featureDesp)
            batch_results.append(results)
        return batch_results

    def predict(self):
        if not self._customizeFeatureSet:
            # generate features
            data, y = self.extract_features()
        else:
            data, y, _featureDesp = self.extract_features(customizeFeatureSet=True)
        return self.predictWithData(data, y)

    def subData(self, data, y):

        newSize = len(data) * 4 / 4
        return data[-newSize:], y[-newSize:],

    def confidenceCal(self, train_data, test_data, predictions, test_y, rf):
        pmax = np.amax(predictions)
        tmax = np.amax(test_y)

        axismax = max(pmax, tmax)

        import forestci as fci
        # calculate inbag and unbiased variance
        inbag = fci.calc_inbag(train_data.shape[0], rf)
        V_IJ, V_IJ_unbiased = fci.random_forest_error(rf, train_data, test_data)

        # print "inbag: {}".format(inbag)
        # print "V_IJ_unbiased: {}".format(V_IJ_unbiased)
        # # Plot error bars for predicted MPG using unbiased variance
        (_, caps, _) = plt.errorbar(predictions, test_y, yerr=np.sqrt(V_IJ), fmt='o', markersize=4, capsize=10,
                                    mfc='red',
                                    mec='green')
        for cap in caps:
            cap.set_markeredgewidth(1)
        plt.title('Error bars for Patient: ' + str(self.patient_id))

        plt.xlabel('Actual BG')
        plt.ylabel('Predicted BG')
        plt.xlim(0, axismax)
        plt.ylim(0, axismax)

        plt.savefig("prediction/tmp/confidence_intervals_bias_patient{}.png".format(self.patient_id))
        plt.close()

        return V_IJ, V_IJ_unbiased

    def predictWithData(self, data, y, _featureDesp="all"):
        labels = y
        y = [label['value'] for label in labels]

        assert (len(data) == len(y))
        # split data
        num_groundtruth = len(data)
        train_size = int(num_groundtruth * self.split_ratio)
        test_size = num_groundtruth - train_size
        # track test instances in original data for access to metadata
        test_glucoseData = self.glucose_data[train_size:]

        # assert (len(test_glucoseData) == test_size)
        # fix train_size, as we ignored the first value
        train_size -= 1
        train_data = data[0:train_size]

        test_glucoseData = labels[train_size:]

        train_y = y[0:train_size]
        test_data = data[train_size:]
        test_y = y[train_size:]

        # assert (len(test_y) == len(test_glucoseData))
        # assert (len(train_y) + len(test_y) + 1 == num_groundtruth)

        rf = None
        if self.tune:
            model = self.models[self.modelName](random_state=30)
            param_grid = self.param_grid
            rf = GridSearchCV(model, param_grid, n_jobs=1, cv=2)
            rf.fit(train_data, train_y)
            self.best_params = rf.best_estimator_;
            self.log.info("Best parameters for patient {} {}".format(self.patient_id, self.best_params))
        else:
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)

        # confident intervals with small data
        # train_data, train_y = self.sub_data(train_data, train_y)
        rf.fit(train_data, train_y)

        predictions = rf.predict(test_data)

        V_IJ, V_IJ_unbiased = self.confidenceCal(train_data, test_data, predictions, test_y, rf)

        # start = datetime.now()

        # predictions = self.feature_select_on_accuracy(train_data, train_y, test_data, rf)

        # print "runtime for feature selection for patient{}. : {}".format(self.patient_id, datetime.now() - start)

        confidence_thrsd = 0.4  # choose from the parameter tuning
        filtered = []
        for i in range(0, len(test_data)):
            if V_IJ[i] >= confidence_thrsd:
                filtered.append(i)

        test_data = np.delete(test_data, filtered)
        test_y = np.delete(test_y, filtered)
        test_glucoseData = np.delete(test_glucoseData, filtered)
        predictions = np.delete(predictions, filtered)

        if len(test_data) == 0:
            return

        # self.confidence_cal(train_data, test_data, test_y, predictions, rf, self.patient_id)

        results = dict()
        results['groundtruth'] = [item['value'] for item in test_glucoseData]
        timestamps = [item['time'] for item in test_glucoseData]
        results['times'] = timestamps
        results['indices'] = [int(item['index']) for item in test_glucoseData]
        results['predictions'] = predictions
        results['performance'] = compute_performance_time_binned(
            timestamps=timestamps,
            groundtruth=test_y,
            predictions=predictions)
        results['performance'].update(compute_performance_meals(
            timestamps=timestamps,
            groundtruth=test_y,
            predictions=predictions,
            carbdata=self.carb_data
        ))
        results['params'] = self.save_params()
        results['featureDesp'] = _featureDesp

        self.plot_learned_model(results['predictions'], results['groundtruth'], results['times'])
        return results

    def feature_select_on_purity(self, train_data, train_y, test_data, rf):
        sfm = SelectFromModel(rf, threshold=0.03)
        sfm.fit(train_data, train_y)

        train_data_imp = sfm.transform(train_data)
        test_data_imp = sfm.transform(test_data)

        rf_imp = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)

        rf_imp.fit(train_data_imp, train_y)

        predictions = rf_imp.predict(test_data_imp)

        return predictions

    def feature_select_on_accuracy(self, train_data, train_y, test_data, rf):
        # rfe = RFE(rf, n_features_to_select=10)

        rfe = RFECV(rf, step=1, cv=5, scoring='neg_mean_squared_error')

        rfe.fit(train_data, train_y)

        train_data_imp = rfe.transform(train_data)

        test_data_imp = rfe.transform(test_data)

        rf_imp = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf)

        rf_imp.fit(train_data_imp, train_y)

        predictions = rf_imp.predict(test_data_imp)

        return predictions
