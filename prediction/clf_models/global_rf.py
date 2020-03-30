from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from util.measures import compute_performance_time_binned, compute_performance_meals, save_confusion_matrix
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from prediction.global_regressor import GlobalRegressor


class GlobalRandomForest(GlobalRegressor):
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
    min_samples_leaf = 2  # small for ExtraTree is helpful


    def __init__(self, modelName):
        super(GlobalRandomForest, self).__init__()
        self.train_X, self.train_Y, self.test_X, self.test_Y, self.test_patient_data, self.test_patient_y, \
        self.test_patient_glucoseData, self.test_patient_carbData, self.classes = self.load_all_data()
        self.model_name = modelName
        if self.model_name == "rf":
            self.min_samples_leaf = 4  # default parameter for Ranfom Forest
            self.n_estimator = 1000

    def save_params(self):
        baseParams = self.save_base_params()
        params = ";".join(("n_estimator: " + str(self.n_estimator), "criterion: " + str(self.criterion),
                           "min_samples_leaf: " + str(self.min_samples_leaf)))
        if self.tune: params = ";".join(
            (baseParams, params, str(self.param_grid), "Best params: " + str(self.best_params)))
        return params


    def predict(self, _feature_desp="all"):
        rf = None
        if self.tune:
            model = ensemble.RandomForestClassifier(random_state=30, class_weight="balanced")
            param_grid = self.param_grid
            rf = GridSearchCV(model, param_grid, n_jobs=-1, cv=2)
            self.best_params = rf.best_estimator_;
            self.log.info("Best parameters: %s")  % self.best_params
        else:
            rf = self.models[self.model_name](n_estimators=self.n_estimator, criterion=self.criterion,
                                              min_samples_leaf=self.min_samples_leaf, class_weight="balanced")

        '''
        Sampling training set
        '''
        print "Original training set shape: {}".format(Counter(self.train_Y))
        print "Original test set shape: {}".format(Counter(self.test_Y))
        self.log.debug("Original training set shape: {}".format(Counter(self.train_Y)))
        # try:
        #     sm = SMOTE(random_state=42, k_neighbors=3)
        #     train_data, train_y = sm.fit_sample(np.asarray(self.train_X), np.asarray(self.train_Y))
        #     self.log.debug("Resampled training set shape: {}".format(Counter(train_y)))
        # except ValueError:
        #     pass
        # rf.fit(train_data, train_y)
        rf.fit(self.train_X, self.train_Y)

        patient_results = dict()
        for patient_id in self.test_patient_data.keys():
            # conduct prediction to patient level
            predictions = rf.predict(self.test_patient_data[patient_id])
            results = self.to_results(test_glucoseData=self.test_patient_glucoseData[patient_id], carbData= self.test_patient_carbData[patient_id], test_y=self.test_patient_y[patient_id],
                                      predictions=predictions, classes=self.classes, patientId=patient_id)
            patient_results[patient_id] = results

        return patient_results

        #Plot non-normalized confusion matrix
        #save_confusion_matrix(cnf_matrix,classes=classes,patient_id=self.patient_id,desc="all",model=self.model_name)


