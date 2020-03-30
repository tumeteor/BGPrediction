from prediction.models.avg import AVG
from prediction.models.lastValue import LastValue
from prediction.models.contextAvg import ContextAVG
from prediction.models.lstm import LSTMs
from prediction.models.rf import RandomForest
from prediction.models.LinearRegression import LR
from prediction.models.arima import ARIMA
from prediction.clf_models.rf import RandomForestClassifier
from prediction.clf_models.SVM import SVM
from prediction.clf_models.Dummy import Dummy
from prediction.clf_models.NaiveBayes import NB
from prediction.clf_models.global_rf import GlobalRandomForest
from prediction.models.FixedModel import FixedModel
from prediction.models.rf_incremental import RandomForest as irf
from prediction.clf_models.crf_incremental import RFClassifier as icc
from ExperimentData import con


### Factory manager class for classifiers ####
class CLFManager:

    @staticmethod
    def factory(patientId=None, dbConnection=con, model=None):
        '''
        :param patientId:
        :param dbConnection:
        :param model:
        :return:
        '''

        '''
        Regression models
        '''
        if model == "avg": return AVG(patientId, dbConnection)
        if model == "last": return LastValue(patientId, dbConnection)
        if model == "contextavg": return ContextAVG(patientId, dbConnection)
        if model == "lstm": return LSTMs(patientId, dbConnection, modelName="lstm")
        if model == "rnn": return LSTMs(patientId, dbConnection, modelName="rnn")
        if model == "rf": return RandomForest(patientId, dbConnection, modelName="rf")
        if model == "et": return RandomForest(patientId, dbConnection, modelName="et")
        if model == "lr": return LR(patientId, dbConnection)
        if model == "arima": return ARIMA(patientId, dbConnection)
        if model == "fm": return FixedModel(patientId, dbConnection)
        if model == "incremental": return irf(patientId, dbConnection, modelName="rf")

        '''
        Classification models
        '''
        if model == "rfc": return RandomForestClassifier(patientId, dbConnection, modelName="rf")
        if model == "etc": return RandomForestClassifier(patientId, dbConnection, modelName="et")
        if model == "svm": return SVM(patientId, dbConnection)
        if model == "dummy": return Dummy(patientId, dbConnection)
        if model == "nb": return NB(patientId, dbConnection)
        if model == "global": return GlobalRandomForest(modelName="rf")
        if model == "icc": return icc(patientId, dbConnection, modelName="rf")
        assert 0, "Bad predictive model creation: " + model
