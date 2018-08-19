from prediction.base_regressor import BaseRegressor
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from Constant import Constant
class RFClassifier(BaseRegressor):

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

    hard_threshold = True

    def __init__(self, patientId, dbConnection, modelName):
        super(RFClassifier, self).__init__(patientId, dbConnection)
        self.modelName = modelName
        if self.modelName == "rf":
            self.min_samples_leaf = 4 # default parameter for Random Forest
            self.n_estimator = 1000

    def saveParams(self):
        baseParams = self.saveBaseParams();
        params = ";".join(("n_estimator: " + str(self.n_estimator), "criterion: " + str(self.criterion), "min_samples_leaf: " + str(self.min_samples_leaf)))
        if self.tune: params = ";".join((baseParams, params, str(self.param_grid), "Best params: " + str(self.best_params)))
        return params

    def predict(self):
        if not self._customizeFeatureSet:
            # generate features
            data, y = self.extractFeatures()
        else:
            data, y, _featureDesp = self.extractFeatures(customizeFeatureSet=True)
        return self.predictWithData(data,y)

    def confidenceCal(self, train_data, test_data, rf):
        import forestci as fci
        from matplotlib import pyplot as plt
        import numpy as np
        # calculate inbag and unbiased variance
        spam_inbag = fci.calc_inbag(train_data.shape[0], rf)
        V_IJ, V_IJ_unbiased = fci.random_forest_error(rf, train_data,
                                                     test_data)

        return V_IJ, V_IJ_unbiased



    def incrementalTrain(self, data, y):
        import pandas as pd
        import datetime

        timestamps = [item['time'] for item in self.glucoseData[1:]]
        assert (len(timestamps) == len(data))


        for i in range (10, len(data)):
            if i % 5 != 0: continue
            rf = self.models[self.modelName](n_estimators=self.n_estimator, criterion=self.criterion,
                                             min_samples_leaf=self.min_samples_leaf, class_weight="balanced")

            #loo = LeaveOneOut()
            kf = KFold(n_splits=5)
            sub_data = data[0:i]
            sub_y = y[0:i]
            sub_ts = np.array(timestamps[0:i])

            p = 0
            total = 0
            tprs = []
            for train_index, test_index in kf.split(sub_data):
                X_train, X_test = sub_data[train_index], sub_data[test_index]
                y_train, y_test = sub_y[train_index], sub_y[test_index]
                ts_train, ts_test = sub_ts[train_index], sub_ts[test_index]

                threshold_variance = 0.04
                y_test_filtered = []
                predictions_filtered = []

                rf.fit(X_train, y_train)
                V_IJ, V_IJ_unbiased = self.confidenceCal(X_train, X_test, rf)
                predictions = rf.predict_proba(X_test)
                # print "variance: {}".format(V_IJ_unbiased)

                assert (len(V_IJ_unbiased) == len(X_test))
                predictions = rf.predict(X_test)


                for j in range(0, len(y_test)):
                     # filter by night time
                     # time = pd.Timestamp(ts_test[j]).tz_localize("UTC")
                     # if time.time() < datetime.time(5):
                         # print time.hour
                         # continue

                     # filter by variance confidence
                     # if V_IJ[j] > threshold_variance: continue
                     # print V_IJ_unbiased[j]
                     # print V_IJ[j]

                     y_test_filtered.append(y_test[j])
                     predictions_filtered.append(predictions[j])
                if len(y_test_filtered) == 0: continue

                for k in range(0, len(y_test_filtered)):
                    if y_test_filtered[k] == 1:
                        total += 1
                    if y_test_filtered[k] == 1 and y_test_filtered[k] == predictions_filtered[k]:
                        p += 1

                if total == 0: continue
                tprs.append(p/float(total))

            print i, np.mean(tprs), np.std(tprs)


    def predictWithData(self, data, Y, _featureDesp="all"):

        if self.patientId != 14: return
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

        self.incrementalTrain(data, np.asarray(cat_Y))


    def tpr(self, cm):
        TN, FP, FN, TP = cm.ravel()
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        return TPR, FDR



