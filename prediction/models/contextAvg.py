from prediction.base_regressor import BaseRegressor
from util.measures import compute_performance_time_binned
from util.measures import compute_performance_meals
import datetime
from util.TimeUtil import tohour
import numpy as np


class ContextAVG(BaseRegressor):
    """
    Baseline predictor that predicts the (weighted) average of previous glucose values observed in a similar (temporal)
    context.
    """
    def __init__(self, patientId, dbConnection):
        super(ContextAVG, self).__init__(patientId, dbConnection)
        # TODO: add global option and possibly use feature manager?
        self.horizon_minutes = 60

    def save_params(self):
        return

    def predict(self):
        """
        Runs ContextAVG value prediction.
        :return:
        """
        assert(self.glucose_data)
        # split the data
        num_groundtruth = len(self.glucose_data)
        train_size = int(num_groundtruth * self.split_ratio)
        test_size = num_groundtruth - train_size
        test_data = self.glucose_data[train_size:]
        assert(len(test_data) == test_size)

        # create prediction list using time weighted avg of previous values
        predictions = list()
        for i in range(0, test_size):
            # time of prediction
            next_time = test_data[i]['time']
            # observed data
            prev_glucose = [item for item in self.glucose_data[:train_size - 1 + i]]
            predictions.append(self.get_time_weighted_average(prev_glucose, next_time))
        assert(len(predictions) == test_size)

        # return ground truth (test set) and predictions (as a dict)
        test_values = [item['value'] for item in test_data]
        results = dict()
        results['groundtruth'] = test_values
        timestamps = [item['time'] for item in test_data]
        results['times'] = timestamps
        results['indices'] = [item['index'] for item in test_data]
        results['predictions'] = predictions
        results['performance'] = compute_performance_time_binned(
            timestamps=timestamps,
            groundtruth=test_values,
            predictions=predictions)
        results['performance'].update(compute_performance_meals(
            timestamps=timestamps,
            groundtruth=test_values,
            predictions=predictions,
            carbdata=self.carb_data
        ))
        results['params'] = None
        return results

    def get_time_weighted_average(self, prev_glucose, next_time):
        """
        :param prev_glucose, next_time:
        :return: average of previous glucose values,
                 rolling average of previous glucose values
        """
        # avg of previous glucose values in same time bin
        # features.append(avg([p['value'] for p in prev_glucose if getTimeBinInt(p['time']) == cur_time_bin]))
        # decreasing weight w/ difference in time
        avg_prev = 0.0
        avg_prev_w = 0.0
        cur_time = next_time - datetime.timedelta(minutes=self.horizon_minutes)
        for prev in prev_glucose:
            if prev['time'] > cur_time:
                continue
            t1 = min(next_time.hour, prev['time'].hour)
            t2 = max(next_time.hour, prev['time'].hour)
            time_diff = min(t2 - t1, t1 + 24 - t2)  # <= 12
            t_w = np.exp(-time_diff)  # weight >.4 for <= 1 hour, >0 only for first 3 hours
            avg_prev += t_w * prev['value']
            avg_prev_w += t_w
        return avg_prev / avg_prev_w