from prediction.base_regressor import BaseRegressor
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from util.measures import compute_performance_time_binned, compute_performance_meals, save_confusion_matrix
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from Constant import Constant


class RandomForestClassifier(BaseRegressor):

    best_params = None

    param_grid = {"n_estimators": [100, 250, 300, 400, 500, 700, 1000],
                  "criterion": ["mae", "mse"],
                  "max_features": ["auto", "sqrt", "log2"],
                  "max_depth": [None, 10, 20]}

    ### Model ####
    # Extratree seems to deal better for small dataset + overfitting
    models = {'rf': ensemble.RandomForestClassifier, 'et': ensemble.ExtraTreesClassifier}

    n_estimator = 1000
    criterion = "gini"
    min_samples_leaf = 2 # small for ExtraTree is helpful

    # classifcation split ratio
    threshold1 = 0.25
    threshold2 = 0.75

    # multiclassification
    multiclass = False

    hard_threshold = False

    def __init__(self, patientId, dbConnection, modelName):
        super(RandomForestClassifier, self).__init__(patientId, dbConnection)
        self.modelName = modelName
        if self.modelName == "rf":
            self.min_samples_leaf = 4 # default parameter for Ranfom Forest
            self.n_estimator = 1000

    def save_params(self):
        base_params = self.save_base_params();
        params = ";".join(("n_estimator: " + str(self.n_estimator), "criterion: " + str(self.criterion), "min_samples_leaf: " + str(self.min_samples_leaf)))
        if self.tune: params = ";".join((base_params, params, str(self.param_grid), "Best params: " + str(self.best_params)))
        return params

    def predict(self):
        if not self._customizeFeatureSet:
            # generate features
            data, y = self.extract_features()
        else:
            data, y, _featureDesp = self.extract_features(customizeFeatureSet=True)
        return self.predict_with_data(data, y)

    @staticmethod
    def confidence_cal(train_data, train_y, test_data, test_y, predictions, rf, patientID):
        import forestci as fci
        from matplotlib import pyplot as plt
        import numpy as np
        # calculate inbag and unbiased variance
        spam_inbag = fci.calc_inbag(train_data.shape[0], rf)
        V_IJ_unbiased = fci.random_forest_error(rf, train_data,
                                                     test_data)

        # Plot forest prediction for emails and standard deviation for estimates
        # Blue points are spam emails; Green points are non-spam emails
        idx = np.where(test_y == 1)[0]
        plt.errorbar(predictions[idx, 1], np.sqrt(V_IJ_unbiased[idx]),
                     fmt='.', alpha=0.75, label='Hyper')

        idx = np.where(test_y == 0)[0]
        plt.errorbar(predictions[idx, 1], np.sqrt(V_IJ_unbiased[idx]),
                     fmt='.', alpha=0.75, label='Non')

        plt.xlabel('Prediction (hyper probability)')
        plt.ylabel('Standard deviation')
        plt.legend()
        plt.show()


    def predict_with_data(self, data, Y, _feature_desp="all"):
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

        rf = None
        if self.tune:
            model = ensemble.RandomForestClassifier(random_state=30, class_weight="balanced")
            param_grid = self.param_grid
            rf = GridSearchCV(model, param_grid, n_jobs=-1, cv=2)
            self.best_params = rf.best_estimator_;
            self.log.info("Best parameters for patient %s %s") % self.patient_id % self.best_params
        else:
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf, class_weight="balanced")

        '''
        Sampling training set
        '''
        print "Original training set shape: {}".format(Counter(train_y))
        print "Original test set shape: {}".format(Counter(test_y))
        self.log.debug("Original training set shape: {}".format(Counter(train_y)))
        # try:
        #     sm = SMOTE(random_state=42, k_neighbors=3)
        #     train_data, train_y = sm.fit_sample(np.asarray(train_data), np.asarray(train_y))
        #     self.log.debug("Resampled training set shape: {}".format(Counter(train_y)))
        # except ValueError:
        #     pass
        rf.fit(train_data, train_y)
        predictions = rf.predict(test_data)

        print "prediction: {}".format(predictions)

        print "accuracy: {}".format(accuracy_score(test_y, predictions, normalize=True))
        print precision_recall_fscore_support(test_y,predictions)

        timestamps = [item['time'] for item in test_glucose_data]
        results = dict()
        results['groundtruth'] = [item['value'] for item in test_glucose_data]
        results['times'] = timestamps
        results['indices'] = [int(item['index']) for item in test_glucose_data]
        results['performance'], results['perClass'] = compute_performance_time_binned(test_y, predictions, timestamps=timestamps,
                                                                                      regression=False, plotConfusionMatrix=False,
                                                                                      classes=classes, patientId=self.patient_id,
                                                                                      model=self.modelName)
        r_meal, r_meal_perclass = compute_performance_meals(test_y, predictions, timestamps=timestamps, plotConfusionMatrix=False,
                                                            classes=classes, patientId=self.patient_id, carbdata=self.carbData, regression=False,
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

        #Plot non-normalized confusion matrix
        save_confusion_matrix(cnf_matrix, classes=classes, patientId=self.patient_id, desc="all", model=self.modelName)


        return results