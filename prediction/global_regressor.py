import ExperimentData
import numpy as np
from Constant import Constant
from prediction.base_regressor import BaseRegressor
import logging
from util.measures import compute_performance_meals, compute_performance_time_binned
from sklearn.metrics import confusion_matrix
class GlobalRegressor(object):

    # classifcation split ratio
    threshold1 = 0.25
    threshold2 = 0.75

    # multiclassification
    multiclass = False

    hard_threshold = False

    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.DEBUG)
        self.log = logging.getLogger("GlobalClassifier")
        self.tune = False
        # parameters
        self.split_ratio = .66
        self.look_back = 8

    def categorized_y(self, y, threshold):
        '''
        Labeling Y
        :param y:
        :param threshold:
        :return:
        '''
        if len(threshold) == 1:
            if y >= threshold[0]:
                return 1
            else:
                return 0

        elif len(threshold) == 2:
            if y <= threshold[0]:
                return 0
            elif y >= threshold[0] and y <= threshold[1]:
                return 1
            else:
                return 2

    def split_train_test(self, data, Y, glucoseData):
        # labeling
        sorted_Y = sorted(Y)  # sort ascending
        thresh1 = int(len(sorted_Y) * self.threshold1) - 1
        thresh2 = int(len(sorted_Y) * self.threshold2) - 1

        '''
        TODO: refactor for code reuse at class siblings
        '''
        if self.hard_threshold:
            cat_Y = [self.categorized_y(y, [Constant.HYPERGLYCEMIA_THRESHOLD]) for y in Y]
            classes = ["non", Constant.HYPER]

        else:
            if self.multiclass:
                cat_Y = [self.categorized_y(y, [sorted_Y[thresh1], sorted_Y[thresh2]]) for y in Y]
                classes = ['low ' + str(self.threshold1 * 100) + '%', 'medium', 'high']
            else:
                cat_Y = [self.categorized_y(y, [sorted_Y[thresh1]]) for y in Y]
                classes = ['low ' + str(self.threshold1 * 100) + '%', 'high']

        assert (len(data) == len(cat_Y))
        # split data
        num_groundtruth = len(glucoseData)
        train_size = int(num_groundtruth * self.split_ratio)
        test_size = num_groundtruth - train_size
        # track test instances in original data for access to metadata
        test_glucose_data = glucoseData[train_size:]
        assert (len(test_glucose_data) == test_size)
        # fix train_size, as we ignored the first value
        train_size -= 1
        train_data = data[0:train_size]
        train_y = cat_Y[0:train_size]
        test_data = data[train_size:]
        test_y = cat_Y[train_size:]
        assert (len(test_y) == len(test_glucose_data))
        assert (len(train_y) + len(test_y) + 1 == num_groundtruth)
        return train_data, train_y, test_data, test_y, classes, test_glucose_data


    def to_results(self, test_glucoseData, carbData, test_y, predictions, classes, patientId):
        timestamps = [item['time'] for item in test_glucoseData]
        results = dict()
        results['groundtruth'] = [item['value'] for item in test_glucoseData]
        results['times'] = timestamps
        results['indices'] = [int(item['index']) for item in test_glucoseData]
        results['performance'], results['perClass'] = compute_performance_time_binned(test_y, predictions,
                                                                                      timestamps=timestamps,
                                                                                      regression=False,
                                                                                      plotConfusionMatrix=False,
                                                                                      classes=classes,
                                                                                      patientId=patientId,
                                                                                      model=self.modelName)
        r_meal, r_meal_perclass = compute_performance_meals(test_y, predictions, timestamps=timestamps,
                                                            plotConfusionMatrix=False,
                                                            classes=classes, patientId=patientId,
                                                            carbdata= carbData, regression=False,
                                                            model=self.modelName)

        results['performance'].update(r_meal)
        results['perClass'].update(r_meal_perclass)

        results['params'] = self.save_params()

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(test_y, predictions)
        np.set_printoptions(precision=2)
        results["report"] = str(cnf_matrix)
        return results

    def load_all_data(self):
        patient_ids = ExperimentData.load_patients()
        train_X = None
        train_Y = None
        test_X = None
        test_Y = None
        test_patient_data = dict()
        test_patient_y = dict()
        test_patient_glucose_data = dict()
        test_patient_carb_data = dict()
        for patient_id in patient_ids:
            base_clf = BaseRegressor(patientId=patient_id, dbConnection=ExperimentData.con)
            data, y = base_clf.extract_features()
            train_data, train_y, test_data, test_y, classes, test_glucoseData = self.split_train_test(data, y, glucoseData=base_clf.glucose_data)
            print train_data.shape
            train_X = np.concatenate((train_data,train_X), axis=0) if not train_X is None else train_data
            train_Y = np.concatenate((train_y, train_Y), axis=0) if not train_Y is None else train_y
            test_X = np.concatenate((test_data, test_X), axis=0) if not test_X is None else test_data
            test_Y = np.concatenate((test_y, test_Y), axis=0) if not test_Y is None else test_y
            test_patient_data[patient_id] = test_data
            test_patient_y[patient_id] = test_y
            test_patient_glucose_data[patient_id] = test_glucoseData
            test_patient_carb_data[patient_id] = base_clf.carbData
        return train_X, train_Y, test_X, test_Y, test_patient_data, test_patient_y, test_patient_glucose_data, test_patient_carb_data, classes

    def save_params(self):
        raise NotImplementedError()

    def save_base_params(self):
        return ";".join(
            ("tune: " + str(self.tune), "look_back: " + str(self.look_back), "split_ratio: " + str(self.split_ratio)))




        


