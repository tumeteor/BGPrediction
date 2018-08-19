__author__ = 'markus'

import logging
from clfManager import CLFManager

import pprint
pp = pprint.PrettyPrinter(indent=2)


class PatientPredictionExperiment:
    """
    Perform single prediction experiment for one user and return prediction results
    """

    # FIXED: Use Factory pattern.
    # Right now we implement here multiple prediction algorithms that have some state in common and differ in
    # other state. It would be best to have subclasses for this. To get an instance of the appropriate class in
    # `Experiment', we should make use of the Factory pattern.
    def __init__(self, patientId=None, algorithm='avg', isBatch=False):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.DEBUG)
        self.log = logging.getLogger("SinglePredictionExperiment")


        # static variables
        self.patientId = patientId
        self.horizon_minutes = 60
        self.algorithm = algorithm

        # batch or single

        self.batch = isBatch

        # mapping of algorithm keys to implementing methods
        self.algorithms = {
            'avg': self.runAvg,
            'last': self.runLastValue,
            'contextavg': self.runContextAvg,
            'lstm': self.runLSTM,
            'rnn': self.runRNN,
            'arima': self.runARIMA,
            'lr': self.runLinearRegression,
            'rf': self.runRandomForest,
            'et': self.runExtraTree,
            'fm': self.runFixedModel,
            'incremental': self.runIncrementalRandomForest
        }


        self.classifer_algorithms = {
            'rf': self.runRandomForestClassifier,
            'svm': self.runSVM,
            'et': self.runExtraTreeClassifier,
            'dummy': self.runDummy,
            'nb': self.runNB,
            'global': self.runGlobal,
            'icc': self.runIncRandomForest
        }
        assert (algorithm in self.algorithms.keys() or algorithm in self.classifer_algorithms.keys())

    def getAlgorithmOptions(self):
        """
        Return keys for implemented algorithms
        """
        return self.algorithms.keys()

    def runClassificationExperiment(self):
        """
        Runs the specified prediction experiment for the given user and returns the results
        :return: A dict containing a list of predictions, the corresponding ground truth values, and patient level
        performance measures TODO: refine
        """
        try:
            return self.classifer_algorithms[self.algorithm]()
        except(TypeError) as e:
            pass

    def runExperiment(self):
        """
        Runs the specified prediction experiment for the given user and returns the results
        :return: A dict containing a list of predictions, the corresponding ground truth values, and patient level
        performance measures TODO: refine
        """

        return self.algorithms[self.algorithm]()


    def runBatchExperiment(self):
        '''
        Run the batch prediction experiment for the given user and return the results
        :return:
        '''
        self.batch = True
        return self.classifer_algorithms[self.algorithm]()

    def runAvg(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Avg training value'
        ))
        return CLFManager.factory(self.patientId, model="avg").predict()

    def runLastValue(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Last value'
        ))
        return CLFManager.factory(self.patientId, model="last").predict()
 
    def runContextAvg(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Weigted AVG in similar context'
        ))
        return CLFManager.factory(self.patientId, model="contextavg").predict()

    def runLinearRegression(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Linear Regression'
        ))
        return CLFManager.factory(self.patientId,model="lr").predict()


    def runRandomForest(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest Regression'
        ))
        if not self.batch:
            results = CLFManager.factory(self.patientId, model="rf").predict()
            return results
        else:
            clf = CLFManager.factory(self.patientId, model="rf")
            return clf.batchPredict(), clf._allFeatureDesp


    def runExtraTree(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Extremely Random Forest Regression'
        ))
        if not self.batch:
            results = CLFManager.factory(self.patientId,model="et").predict()
            return results
        else:
            clf = CLFManager.factory(self.patientId,model="et")
            return clf.batchPredict(), clf._allFeatureDesp

    def runIncrementalRandomForest(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest Regression'
        ))
        results = CLFManager.factory(self.patientId, model="incremental").predict()
        return results


    def runARIMA(self):
        """
        Runs ARIMA prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'ARIMA'
        ))
        arima = CLFManager.factory(self.patientId,model="arima")

        return arima.predict()


    def runLSTM(self):
        """
        Runs LSTM prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'LSTM-Recurrent Neural Networks'
        ))
        lstm = CLFManager.factory(self.patientId,model="lstm")

        return lstm.predict()

    def runRNN(self):
        """
        Runs RNN prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Recurrent Neural Networks'
        ))
        rnn = CLFManager.factory(self.patientId,model="rnn")

        return rnn.predict()


    def runFixedModel(self):
        """
        Runs RNN prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'FixedModel'
        ))
        fixedModel = CLFManager.factory(self.patientId,model="fm")

        return fixedModel.predict()

    def runRandomForestClassifier(self):
        '''
        Run Random Forest Classifier
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest'
        ))
        return CLFManager.factory(self.patientId,model="rfc").predict()


    def runExtraTreeClassifier(self):
        '''
        Run Extr TRee Classifier
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId,  'Extremely Random Forest'
        ))
        return CLFManager.factory(self.patientId,model="etc").predict()

    def runDummy(self):
        '''
        Run Dummy classifier
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Dummy'
        ))
        return CLFManager.factory(self.patientId,model="dummy").predict()

    def runSVM(self):
        '''
        Run SVM
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'SVM'
        ))
        return CLFManager.factory(self.patientId,model="svm").predict()

    def runNB(self):
        '''
        Run Naive Bayes
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Naive Bayes'
        ))
        return CLFManager.factory(self.patientId,model="nb").predict()


    def runGlobal(self):
        '''
        Run global model
        :return:
        '''
        self.log.info("Running global classification using algorithm '{}'".format(
            self.patientId, 'Naive Bayes'
        ))
        return CLFManager.factory(model="global").predict()

    def runIncRandomForest(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest Classifier'
        ))
        results = CLFManager.factory(self.patientId, model="icc").predict()
        return results








