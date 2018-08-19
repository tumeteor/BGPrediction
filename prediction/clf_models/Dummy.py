from prediction.base_regressor import BaseRegressor
from util.measures import computePerformanceTimeBinned, computePerformanceMeals
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,accuracy_score
import numpy as np
from util.measures import save_confusion_matrix
from sklearn.dummy import DummyClassifier
from Constant import Constant

class Dummy(BaseRegressor):

    modelName = "dummy"

    # classifcation split ratio
    threshold1 = 0.25
    threshold2 = 0.75

    # multiclassification
    multiclass = False

    strategy = "most_frequent"
    random_state = 0


    hard_threshold = False

    def __init__(self, patientId, dbConnection):
        super(Dummy, self).__init__(patientId, dbConnection)


    def saveParams(self):
        baseParams = self.saveBaseParams();
        params = ";".join(("strategy: " + str(self.strategy),"random_state: " + str(self.random_state)))
        return params

    def predict(self):
        # generate features
        data, y = self.extractFeatures()
        return self.predictWithData(data,y)


    def predictWithData(self, data, Y, _featureDesp="all"):
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
        num_groundtruth = len(self.glucoseData)
        train_size = int(num_groundtruth * self.split_ratio)
        test_size = num_groundtruth - train_size
        # track test instances in original data for access to metadata
        test_glucoseData = self.glucoseData[train_size:]
        assert (len(test_glucoseData) == test_size)
        # fix train_size, as we ignored the first value
        train_size -= 1
        train_data = data[0:train_size]
        train_y = cat_Y[0:train_size]
        test_data = data[train_size:]
        test_y = cat_Y[train_size:]
        assert (len(test_y) == len(test_glucoseData))
        assert (len(train_y) + len(test_y) + 1 == num_groundtruth)

        clf = DummyClassifier(strategy=self.strategy, random_state=self.random_state)
        clf.fit(train_data, train_y) # ignore train_data
        predictions = clf.predict(test_data)

        print "prediction: {}".format(predictions)

        print "accuracy: {}".format(accuracy_score(test_y, predictions, normalize=True))
        print precision_recall_fscore_support(test_y, predictions)

        timestamps = [item['time'] for item in test_glucoseData]
        results = dict()
        results['performance'], results['perClass'] = computePerformanceTimeBinned(test_y, predictions,
                                                                                   timestamps=timestamps,
                                                                                   regression=False,
                                                                                   plotConfusionMatrix=True,
                                                                                   classes=classes,
                                                                                   patientId=self.patientId,
                                                                                   model=self.modelName)
        r_meal, r_meal_perclass = computePerformanceMeals(test_y, predictions, timestamps=timestamps,
                                                          plotConfusionMatrix=True,
                                                          classes=classes, patientId=self.patientId,
                                                          carbdata=self.carbData, regression=False,
                                                          model=self.modelName)

        results['performance'].update(r_meal)
        results['perClass'].update(r_meal_perclass)

        results['params'] = self.saveParams()

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(test_y, predictions)
        np.set_printoptions(precision=2)
        results["report"] = "Binary classification with label: {}".format(classes)
        results["report"] = "number of instance in {}: {} and in {}: {}".format(classes[0], thresh1 + 1,
                                                                                classes[1], len(Y) - thresh1 - 1)
        results["report"] += ";confusion matrix: " + str(cnf_matrix)

        # Plot non-normalized confusion matrix
        save_confusion_matrix(cnf_matrix, classes=classes, patientId=self.patientId, desc="all", model=self.modelName)

        return results



