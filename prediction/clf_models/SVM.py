from prediction.base_regressor import BaseRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from util.measures import compute_performance_time_binned, compute_performance_meals
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
from util.measures import save_confusion_matrix


class SVM(BaseRegressor):

    best_params = None

    param_grid = {"kernel": ["linear","rbf"],
                  "C": [1, 10]}
    kernel = "rbf"
    C = 1

    modelName = "svm"

    # classifcation split ratio
    threshold1 = 0.25
    threshold2 = 0.75

    # multiclassification
    multiclass = False


    def __init__(self, patientId, dbConnection):
        super(SVM, self).__init__(patientId, dbConnection)


    def save_params(self):
        base_params = self.save_base_params();
        params = ";".join(("kernel: " + str(self.kernel),"C: " + str(self.C)))
        if self.tune: params = ";".join((base_params, params, str(self.param_grid), "Best params: " + str(self.best_params)))
        return params

    def predict(self):
        if not self._customizeFeatureSet:
            # generate features
            data, y = self.extract_features()
        else:
            data, y, _featureDesp = self.extract_features(customizeFeatureSet=True)
        return self.predict_with_data(data, y)


    def predict_with_data(self, data, Y, _feature_desp="all"):
        # labeling
        sorted_Y = sorted(Y)  # sort ascending
        thresh1 = int(len(sorted_Y) * self.threshold1) - 1
        thresh2 = int(len(sorted_Y) * self.threshold2) - 1

        if self.multiclass:
            cat_Y = [self.categorized_y(y, [sorted_Y[thresh1], sorted_Y[thresh2]]) for y in Y]
            classes = ['low ' + str(self.threshold1 * 100) + '%', 'medium', 'high']

        else:
            cat_Y = [self.categorized_y(y, [sorted_Y[thresh1]]) for y in Y]
            classes = ['low ' + str(self.threshold1 * 100) + '%', 'high']

        assert (len(data) == len(cat_Y))
        # split data
        num_groundtruth = len(self.glucose_data)
        train_size = int(num_groundtruth * self.split_ratio)
        test_size = num_groundtruth - train_size
        # track test instances in original data for access to metadata
        test_glucose_data = self.glucose_data[train_size:]
        assert (len(test_glucose_data) == test_size)
        # fix train_size, as we ignored the first value
        train_size -= 1
        train_data = data[0:train_size]
        train_y = cat_Y[0:train_size]
        test_data = data[train_size:]
        test_y = cat_Y[train_size:]
        assert (len(test_y) == len(test_glucose_data))
        assert (len(train_y) + len(test_y) + 1 == num_groundtruth)

        clf = None
        predictions = None
        if self.tune:
            clf = svm.SVC()
            grid = GridSearchCV(clf, param_grid=self.param_grid, cv=5, refit=True, scoring='accuracy')
            grid.fit(train_data, train_y)
            self.best_params = grid.best_estimator_;
            self.log.info("Best parameters for patient {} {}".format(self.patient_id, self.best_params))
            predictions = grid.predict(test_data)
        else:
            clf = svm.SVC()
            clf.fit(train_data, train_y)
            print train_y
            print predictions
            predictions = clf.predict(test_data)

        print precision_recall_fscore_support(test_y, predictions, average='weighted')
        print "accuracy: {}".format(accuracy_score(test_y, predictions, normalize=False))

        timestamps = [item['time'] for item in test_glucose_data]
        results = dict()
        results['performance'], results['perClass'] = compute_performance_time_binned(test_y, predictions,
                                                                                      timestamps=timestamps,
                                                                                      regression=False,
                                                                                      plotConfusionMatrix=True,
                                                                                      classes=classes,
                                                                                      patientId=self.patient_id,
                                                                                      model=self.modelName)
        r_meal, r_meal_perclass = compute_performance_meals(test_y, predictions, timestamps=timestamps,
                                                            plotConfusionMatrix=True,
                                                            classes=classes, patientId=self.patient_id,
                                                            carbdata=self.carb_data, regression=False,
                                                            model=self.modelName)


        results['performance'].update(r_meal)
        results['perClass'].update(r_meal_perclass)

        results['params'] = self.save_params()

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(test_y, predictions)
        np.set_printoptions(precision=2)
        results["report"] = "Binary classification with label: {}".format(classes)
        results["report"] = "number of instance in {}: {} and in {}: {}".format(classes[0], thresh1 + 1,
                                                                                classes[1], len(Y) - thresh1 - 1)
        results["report"] += ";confusion matrix: " + str(cnf_matrix)

        # Plot non-normalized confusion matrix
        save_confusion_matrix(cnf_matrix, classes=classes, patientId=self.patient_id, desc="all", model=self.modelName)

        return results



