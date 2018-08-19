from prediction.base_regressor import BaseRegressor
from util.measures import computePerformanceTimeBinned
from util.measures import computePerformanceMeals
from sklearn import linear_model
class LR(BaseRegressor):

    def __init__(self, patientId, dbConnection):
        super(LR, self).__init__(patientId, dbConnection)

    def saveParams(self):
        return self.saveBaseParams();

    def batchPredict(self):
        '''
        batch experiment for different feature subsets
        :return:
        '''
        X, y = self.Features.buildFeatureMatrix(self.look_back)
        _allSubsets = self.Features.FeatureGroupAllSubsets(X)
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
            data, y = self.extractFeatures()
        else:
            data, y, _featureDesp = self.extractFeatures(customizeFeatureSet=True)
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
        test_glucoseData = self.glucoseData[train_size:]


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

        # self.confidenceCal(train_data, test_data, test_y, predictions, rf, self.patientId)

        results = dict()
        results['groundtruth'] = [item['value'] for item in test_glucoseData]
        timestamps = [item['time'] for item in test_glucoseData]
        results['times'] = timestamps
        results['indices'] = [int(item['index']) for item in test_glucoseData]
        results['predictions'] = predictions
        results['performance'] = computePerformanceTimeBinned(
            timestamps=timestamps,
            groundtruth=test_y,
            predictions=predictions)
        results['performance'].update(computePerformanceMeals(
            timestamps=timestamps,
            groundtruth=test_y,
            predictions=predictions,
            carbdata=self.carbData
        ))
        results['params'] = self.saveParams()
        results['featureDesp'] = _featureDesp

        self.plotLearnedModel(results['predictions'],results['groundtruth'],results['times'])
        return results

