from prediction.base_regressor import BaseRegressor
from util.measures import compute_performance_time_binned
from util.measures import compute_performance_meals
from sklearn import linear_model
class LR(BaseRegressor):

    def __init__(self, patientId, dbConnection):
        super(LR, self).__init__(patientId, dbConnection)

    def save_params(self):
        return self.save_base_params();

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
            results = self.predictWithData(data,y,featureDesp)
            self._allFeatureDesp.append(featureDesp)
            batch_results.append(results)
        return batch_results

    def predict(self):
        if not self._customizeFeatureSet:
            # generate features
            data, y = self.extract_features()
        else:
            data, y, _featureDesp = self.extract_features(customizeFeatureSet=True)
        return self.predictWithData(data,y)


    def subData(self, data, y):

        newSize = len(data) * 4/4
        return data[-newSize:], y[-newSize:],


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


        #assert (len(test_glucoseData) == test_size)
        # fix train_size, as we ignored the first value
        train_size -= 1
        train_data = data[0:train_size]

        test_glucoseData = labels[train_size:]


        train_y = y[0:train_size]
        test_data = data[train_size:]
        test_y = y[train_size:]

        lr = linear_model.LinearRegression()

        lr.fit(train_data, train_y)

        predictions = lr.predict(test_data)



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
            carbdata=self.carbData
        ))
        results['params'] = self.save_params()
        results['featureDesp'] = _featureDesp

        self.plot_learned_model(results['predictions'], results['groundtruth'], results['times'])
        return results

