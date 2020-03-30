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
    def __init__(self, patient_id=None, algorithm='avg', is_batch=False):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.DEBUG)
        self.log = logging.getLogger("SinglePredictionExperiment")

        # static variables
        self.patientId = patient_id
        self.horizon_minutes = 60
        self.algorithm = algorithm

        # batch or single

        self.batch = is_batch

        # mapping of algorithm keys to implementing methods
        self.algorithms = {
            'avg': self.run_avg,
            'last': self.run_last_value,
            'contextavg': self.run_context_avg,
            'lstm': self.run_lstm,
            'rnn': self.run_rnn,
            'arima': self.run_arima,
            'lr': self.run_linear_regression,
            'rf': self.run_random_forest,
            'et': self.run_extra_tree,
            'fm': self.run_fixed_model,
            'incremental': self.run_incremental_random_forest
        }

        self.classifer_algorithms = {
            'rf': self.run_random_forest_classifier,
            'svm': self.run_svm,
            'et': self.run_extra_tree_classifier,
            'dummy': self.run_dummy,
            'nb': self.run_nb,
            'global': self.run_global,
            'icc': self.run_inc_random_forest
        }
        assert (algorithm in self.algorithms.keys() or algorithm in self.classifer_algorithms.keys())

    def get_algorithm_options(self):
        """
        Return keys for implemented algorithms
        """
        return self.algorithms.keys()

    def run_classification_experiment(self):
        """
        Runs the specified prediction experiment for the given user and returns the results
        :return: A dict containing a list of predictions, the corresponding ground truth values, and patient level
        performance measures TODO: refine
        """
        try:
            return self.classifer_algorithms[self.algorithm]()
        except(TypeError) as e:
            pass

    def run_experiment(self):
        """
        Runs the specified prediction experiment for the given user and returns the results
        :return: A dict containing a list of predictions, the corresponding ground truth values, and patient level
        performance measures TODO: refine
        """

        return self.algorithms[self.algorithm]()

    def run_batch_experiment(self):
        '''
        Run the batch prediction experiment for the given user and return the results
        :return:
        '''
        self.batch = True
        return self.classifer_algorithms[self.algorithm]()

    def run_avg(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Avg training value'
        ))
        return CLFManager.factory(self.patientId, model="avg").predict()

    def run_last_value(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Last value'
        ))
        return CLFManager.factory(self.patientId, model="last").predict()

    def run_context_avg(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Weigted AVG in similar context'
        ))
        return CLFManager.factory(self.patientId, model="contextavg").predict()

    def run_linear_regression(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Linear Regression'
        ))
        return CLFManager.factory(self.patientId, model="lr").predict()

    def run_random_forest(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest Regression'
        ))
        if not self.batch:
            results = CLFManager.factory(self.patientId, model="rf").predict()
            return results
        else:
            clf = CLFManager.factory(self.patientId, model="rf")
            return clf.batchPredict(), clf._allFeatureDesp

    def run_extra_tree(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Extremely Random Forest Regression'
        ))
        if not self.batch:
            results = CLFManager.factory(self.patientId, model="et").predict()
            return results
        else:
            clf = CLFManager.factory(self.patientId, model="et")
            return clf.batchPredict(), clf._allFeatureDesp

    def run_incremental_random_forest(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest Regression'
        ))
        results = CLFManager.factory(self.patientId, model="incremental").predict()
        return results

    def run_arima(self):
        """
        Runs ARIMA prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'ARIMA'
        ))
        arima = CLFManager.factory(self.patientId, model="arima")

        return arima.predict()

    def run_lstm(self):
        """
        Runs LSTM prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'LSTM-Recurrent Neural Networks'
        ))
        lstm = CLFManager.factory(self.patientId, model="lstm")

        return lstm.predict()

    def run_rnn(self):
        """
        Runs RNN prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Recurrent Neural Networks'
        ))
        rnn = CLFManager.factory(self.patientId, model="rnn")

        return rnn.predict()

    def run_fixed_model(self):
        """
        Runs RNN prediction.
        :return:
        """
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'FixedModel'
        ))
        fixed_model = CLFManager.factory(self.patientId, model="fm")

        return fixed_model.predict()

    def run_random_forest_classifier(self):
        '''
        Run Random Forest Classifier
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest'
        ))
        return CLFManager.factory(self.patientId, model="rfc").predict()

    def run_extra_tree_classifier(self):
        '''
        Run Extr TRee Classifier
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Extremely Random Forest'
        ))
        return CLFManager.factory(self.patientId, model="etc").predict()

    def run_dummy(self):
        '''
        Run Dummy classifier
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Dummy'
        ))
        return CLFManager.factory(self.patientId, model="dummy").predict()

    def run_svm(self):
        '''
        Run SVM
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'SVM'
        ))
        return CLFManager.factory(self.patientId, model="svm").predict()

    def run_nb(self):
        '''
        Run Naive Bayes
        :return:
        '''
        self.log.info("Running classification for patient {} using algorithm '{}'".format(
            self.patientId, 'Naive Bayes'
        ))
        return CLFManager.factory(self.patientId, model="nb").predict()

    def run_global(self):
        '''
        Run global model
        :return:
        '''
        self.log.info("Running global classification using algorithm '{}'".format(
            self.patientId, 'Naive Bayes'
        ))
        return CLFManager.factory(model="global").predict()

    def run_inc_random_forest(self):
        self.log.info("Running prediction for patient {} using algorithm '{}'".format(
            self.patientId, 'Random Forest Classifier'
        ))
        results = CLFManager.factory(self.patientId, model="icc").predict()
        return results
