import time
import logging
import pandas as pd
from feature_manager import FeatureManager
import numpy as np
import matplotlib.dates as md


class BaseRegressor(object):

    def __init__(self, patientId, dbConnection, plotFeatureImportance=False):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.DEBUG)
        self.log = logging.getLogger("BaseClassifier")
        self.patient_id = patientId
        self.con = dbConnection

        self.glucose_data = list()
        self.insulinData = list()
        self.carbData = list()
        self.activityData = list()
        # load data necessary for ARIMA
        self.discretizedData = list()

        ###### LOAD DATA ######
        ###### do not change line order ###
        self.load_all_data()

        ###### LOAD Feature Extraction ####
        self.Features = FeatureManager(self.glucose_data, self.insulinData, self.carbData,
                                       self.activityData, self.patient_id)
        # tuning option for RF
        # set it now as a common parameter
        # for all models
        self.tune = False
        # parameters
        self.split_ratio = .66
        self.look_back = 8
        self._plotFeatureImportance = plotFeatureImportance
        self._plotLearnedModel = False

        # customize feature set option
        # TODO: set from outside
        self._customizeFeatureSet = False

        self._allFeatureDesp = list()

    def load_all_data(self):
        ###### LOAD DATA ######
        self.load_glucose_data()
        self.load_insulin_data()
        self.load_carb_data()
        self.load_activity_data()
        # load data necessary for ARIMA
        self.load_discretized_data()

    def load_glucose_data(self):
        """
        Retrieve glucose (ground truth) data from database
        """
        self.log.info("Loading Glucose data for patient {}".format(self.patient_id))
        with self.con:
            cur = self.con.cursor()
            query = "SELECT date as 'time', `gt-value` as 'value', pos as 'index' FROM BG_Instance " \
                    "WHERE patientID = {patientId} and date > '2017-02-25'".format(patientId=self.patient_id)
            self.log.debug("load_glucose_data() query: '" + query + "'")
            cur.execute(query)
            logging.debug("{} rows returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No Glucose data was returned!")
                return
            for row in rows:
                self.glucose_data.append(row)

        logging.debug("{} glucose measurements returned".format(len(self.glucose_data)))

    def load_insulin_data(self, ignoreBasal=False):
        """
        Retrieve insulin data
        """
        self.log.info("Loading insulin data for patient {}".format(self.patient_id))
        if ignoreBasal:
            with self.con:
                cur = self.con.cursor()
                query = "SELECT date as 'time', value, type FROM BG_Insulin " \
                        "WHERE patientID = {patientId} and date > '2017-02-25' and type='rapid'".format(
                    patientId=self.patient_id)
                self.log.debug("load_insulin_data() query: '" + query + "'")
                cur.execute(query)
                logging.debug("{} rows returned".format(cur.rowcount))
                rows = cur.fetchall()
                if not rows:
                    self.log.error("No insulin data was returned!")
                    return
                for row in rows:
                    self.insulinData.append(row)
        else:
            with self.con:
                cur = self.con.cursor()
                query = "SELECT date as 'time', value, type FROM BG_Insulin " \
                        "WHERE patientID = {patientId} and date > '2017-02-25'".format(patientId=self.patient_id)
                self.log.debug("load_insulin_data() query: '" + query + "'")
                cur.execute(query)
                logging.debug("{} rows returned".format(cur.rowcount))
                rows = cur.fetchall()
                if not rows:
                    self.log.error("No insulin data was returned!")
                    return
                for row in rows:
                    self.insulinData.append(row)

    def load_carb_data(self):
        """
        Retrieve carbohydrate data
        """
        self.log.info("Loading carbohydrate data for patient {}".format(self.patient_id))
        with self.con:
            cur = self.con.cursor()
            query = "SELECT date as 'time', value FROM BG_carbohydrate " \
                    "WHERE patientID = {patientId} and date > '2017-02-25'".format(patientId=self.patient_id)
            self.log.debug("loadCarbohydrateData() query: '" + query + "'")
            cur.execute(query)
            logging.debug("{} rows returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No carb data was returned!")
                return
            for row in rows:
                self.carbData.append(row)

    def load_activity_data(self):
        """
        Retrieve activity data
        """
        # FIXED: import steps and use them in place of Akcal
        self.log.info("Loading activity data for patient {}".format(self.patient_id))
        with self.con:
            cur = self.con.cursor()
            query = "SELECT date as 'time', value FROM BG_steps " \
                    "WHERE patientID = {patientId} and date > '2017-02-25'".format(patientId=self.patient_id)
            self.log.debug("load_activity_data() query: '" + query + "'")
            cur.execute(query)
            logging.debug("{} rows returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No activity data was returned!")
                return
            for row in rows:
                self.activityData.append(row)

    def load_discretized_data(self):
        self.log.info("Loading discretized glucose data for patient {}".format(self.patient_id))
        with self.con:
            cur = self.con.cursor()
            query = "SELECT date, bloodglucose as 'bg' FROM BG_Timeseries " \
                    "WHERE patientID = {patient_id} and date > '2017-02-25' ".format(patientId=self.patient_id)
            df = pd.read_sql(query, self.con)
            logging.debug("{} rows returned".format(len(df)))
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        # IMPORTANT: sort by date
        df = df.sort_index()

        # select the time only from 2017-03-01
        # where we have stable measurement data
        df = df["2017-02-25":]
        min_date = min(df['bg'].index)
        max_date = max(df['bg'].index)
        df = df[min_date:max_date]
        self.discretizedData = df.values

    ''''' load the raw timestamp for blood glucose data '''

    def load_timestamps(self, con, patientId):
        with con:
            cur = con.cursor()
            query = "SELECT date, pos FROM BG_Instance " \
                    "WHERE patientID = {patientId} ".format(patientId=patientId)
            df = pd.read_sql(query, con)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        # IMPORTANT: sort by date
        df = df.sort_index()

        return df

    def predict(self):
        raise NotImplementedError()

    def save_base_params(self):
        return ";".join(
            ("tune: " + str(self.tune), "look_back: " + str(self.look_back), "split_ratio: " + str(self.split_ratio)))

    def save_params(self):
        raise NotImplementedError()

    def extract_features(self, customizeFeatureSet=False, customGroup=None):
        X, Y = self.Features.build_feature_matrix(self.look_back)
        if customGroup != None:
            return self.Features.custom_feature_group(X, customGroup), Y
        if not customizeFeatureSet:
            return X, Y
        else:
            new_X, desp = self.Features.customFeatureGroupSubset(X)
            return new_X, Y, desp

    def to_result(self, test_glucoseData, predictions, test_y, timestamps):
        """
        :param test_glucoseData:
        :param predictions:
        :param test_y:
        :param timestamps:
        :return:
        """
        pass
        # TODO: return ground truth (test set) and predictions (as a dict)
        results = dict()
        return results

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

    def select_k_importance(self, model, X, k=10):
        """
        Only for Decision Tree-based models
        :param model:
        :param X:
        :param k:
        :return:
        """
        return X[:, model.feature_importances_.argsort()[::-1][:k]]

    def plot_feature_importance(self, model, X):
        """
        Only for Decision Tree-based models
        :param model:
        :param X:
        :return:
        """
        if not self._plotFeatureImportance:
            return
        # import matplotlib only when necessary
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True, 'font.size': 10})
        # do the plotting
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.barh(range(X.shape[1]), importances[indices],
                 color="r", xerr=std[indices], align="center")
        # If you want to define your own labels,
        # change indices to a list of labels on the following line.
        plt.yticks(range(X.shape[1]), self.Features.featureNames)
        plt.ylim([-1, X.shape[1]])
        plt.savefig("prediction/tmp/feature_importance{}.png".format(self.patient_id))
        plt.close()

    def plot_learned_model(self, test, sample, timestamp):
        '''

        :param test:
        :param sample:
        :return:
        '''
        # import matplotlib only when necessary
        import matplotlib.pyplot as plt
        if not self._plotLearnedModel: return

        assert len(test) == len(sample)
        xfmt = md.DateFormatter('%Y-%m-%d %H')

        plt.figure()
        plt.title("Prediction Variance")
        ax = plt.gca()
        ax.set_xticks(timestamp)
        ax.xaxis.set_major_formatter(xfmt)
        p = plt.plot(timestamp, test, label="prediction")
        s = plt.plot(timestamp, sample, label="sample")
        plt.legend()
        plt.savefig("prediction/tmp/prediction_time{}.png".format(self.patient_id))
        plt.close()
