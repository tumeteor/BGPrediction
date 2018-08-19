from prediction.base_regressor import BaseRegressor
from util.measures import computePerformanceTimeBinned
from util.measures import computePerformanceMeals


class LastValue(BaseRegressor):
    """
    Baseline predictor that always predicts the last observed value.
    """
    def __init__(self, patientId, dbConnection):
        super(LastValue, self).__init__(patientId, dbConnection)

    def saveParams(self):
        return

    def predict(self):
        """
        Runs last value prediction.
        :return:
        """
        assert(self.glucoseData)
        # split the data
        num_groundtruth = len(self.glucoseData)
        train_size = int(num_groundtruth * self.split_ratio)
        test_size = num_groundtruth - train_size
        train_data = self.glucoseData[0:train_size]
        test_data = self.glucoseData[train_size:]
        assert(len(test_data) == test_size)
        # compute avg on training data
        last_value = train_data[-1]['value']
        # create prediction list using avg
        test_values = [item['value'] for item in test_data]
        predictions = list()
        for i in range(0, test_size):
            predictions.append(last_value)
            last_value = test_values[i]
        assert(len(predictions) == test_size)
        # return ground truth (test set) and predictions (as a dict)
        results = dict()
        results['groundtruth'] = test_values
        timestamps = [item['time'] for item in test_data]
        results['times'] = timestamps
        results['indices'] = [item['index'] for item in test_data]
        results['predictions'] = predictions
        results['performance'] = computePerformanceTimeBinned(
            timestamps=timestamps,
            groundtruth=test_values,
            predictions=predictions)
        results['performance'].update(computePerformanceMeals(
            timestamps=timestamps,
            groundtruth=test_values,
            predictions=predictions,
            carbdata=self.carbData
        ))
        results['params'] = None

        return results