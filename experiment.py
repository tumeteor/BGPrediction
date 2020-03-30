__author__ = 'tu and markus'

import logging
from argparse import ArgumentParser

from prediction.patient_experiment import PatientPredictionExperiment
from util.measures import compute_performance_metrics
from collections import defaultdict
import ExperimentData

import pprint
from Constant import Constant

pp = pprint.PrettyPrinter(indent=2)


class PredictionExperiment:
    """
    Perform prediction experiment for all users and store prediction results
    """

    def __init__(self, algorithm='avg', svn_rev=0):
        # self.readonly = cfg.data['readonly']
        # configure log
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
        self.log = logging.getLogger("PredictionExperiment")
        self.algorithm = algorithm
        self.svn_rev = svn_rev

        # patient data
        self.patientIDs = ExperimentData.load_patients()
        # batch id for running experiments in batch
        self.isBatch = False
        self.batchId = 0
        self._allFeatureDesp = None

        self.con = ExperimentData.con

        # print info on experiment
        self.log.info("Set up Experiment for algorithm '{}'".format(
            self.algorithm
        ))

        # learning mode
        self.regression = True

    def run_clf_experiment(self):
        '''
        run classification experiment
        :return:
        '''
        results_single = dict()
        for patient in self.patientIDs:
            experiment = PatientPredictionExperiment(
                patient_id=patient, algorithm=self.algorithm,
                is_batch=None
            )
            results_single[patient] = experiment.run_classification_experiment()

        self.experiment_id = self.store_experiment(mode='c', featureDesp='all')
        for patient in self.patientIDs:
            for subset in results_single[patient]['performance']:
                self.store_clf_results_subset(results_single[patient], patient, subset)

        results_overall = self.process_results(results_single)
        self.store_aggregate_results(results_overall)

    def run_global_clf_experiment(self):
        '''
        run classification experiment
        :return:
        '''
        experiment = PatientPredictionExperiment(algorithm=self.algorithm,
                                                 is_batch=None
                                                 )
        results_single = experiment.run_classification_experiment()
        self.experiment_id = self.store_experiment(mode='c', featureDesp='all')
        for patient in self.patientIDs:
            for subset in results_single[patient]['performance']:
                self.store_clf_results_subset(results_single[patient], patient, subset)

        results_overall = self.process_results(results_single)
        self.store_aggregate_results(results_overall)

        # self.experiment_id = self.store_experiment(mode='c', featureDesp='all')

    def store_clf_results_subset(self, result_single, patientId, subset):
        '''
        store classification results
        :param result_single:
        :param patientId:
        :return:
        '''

        self.log.info("Storing Classification Experiment for patient {} at {}".format(patientId, subset))
        results = result_single['performance'][subset]
        with self.con:
            cur = self.con.cursor()
            query = None
            # TODO: report for all subsets?
            if subset == "all":
                query = "INSERT IGNORE INTO BG_clf_patient (experiment_run_id, subset, patientID, num_values, accuracy, `precision`, recall, f1, params, features, report)" \
                        "VALUES (%(experiment_run_id)s, %(subset)s, %(patientID)s, %(num_values)s, %(accuracy)s, %(precision)s, %(recall)s, %(f1)s, %(params)s, %(features)s, %(report)s)"
                cur.execute(query, {
                    "experiment_run_id": self.experiment_id,
                    "subset": subset,
                    "patientID": patientId,
                    "num_values": results['num_values'],
                    "accuracy": results['accuracy'],
                    "precision": results['precision'],
                    "recall": results['recall'],
                    "f1": results['f1'],
                    "params": result_single['params'],
                    "features": "all",
                    "report": result_single['report']
                })
            else:
                query = "INSERT IGNORE INTO BG_clf_patient (experiment_run_id, subset, patientID, num_values, accuracy, `precision`, recall, f1, params, features)" \
                        "VALUES (%(experiment_run_id)s, %(subset)s, %(patientID)s, %(num_values)s, %(accuracy)s, %(precision)s, %(recall)s, %(f1)s, %(params)s, %(features)s)"

                self.log.debug("storeClfResutls: {}".format(query))
                cur.execute(query, {
                    "experiment_run_id": self.experiment_id,
                    "subset": subset,
                    "patientID": patientId,
                    "num_values": results['num_values'],
                    "accuracy": results['accuracy'],
                    "precision": results['precision'],
                    "recall": results['recall'],
                    "f1": results['f1'],
                    "params": result_single['params'],
                    "features": "all"
                })
            for rkey in result_single['perClass'][subset]:
                r = result_single['perClass'][subset][rkey]
                query = "INSERT IGNORE INTO BG_cm_patient (experiment_run_id, subset, patientID, label, noInstances, tp, tn, fp, fn, accuracy, `precision`, recall, f1)" \
                        "VALUES (%(experiment_run_id)s, %(subset)s, %(patientID)s, %(label)s, %(noInstances)s, %(tp)s, %(tn)s, %(fp)s, %(fn)s, %(accuracy)s, %(precision)s, %(recall)s, %(f1)s)"
                cur.execute(query, {
                    "experiment_run_id": self.experiment_id,
                    "subset": subset,
                    "patientID": patientId,
                    "label": r['className'],
                    "noInstances": r['noInstances'],
                    "tp": r['tp'],
                    "tn": r['tn'],
                    "fp": r['fp'],
                    "fn": r['fn'],
                    "accuracy": r['accuracy'],
                    "precision": r['precision'],
                    "recall": r['recall'],
                    "f1": r['f1']
                })

    def run_regression_experiments(self):
        """
        Run prediction experiments for the given algorithm and settings for each of the patients.
        :return:
        """
        assert self.patientIDs  # assert not empty
        self.log.info("Running experiments for {} patients".format(len(self.patientIDs)))

        results_batch = defaultdict(dict)
        results_single = dict()
        for patient in self.patientIDs:
            patient_experiment = PatientPredictionExperiment(
                patient_id=patient, algorithm=self.algorithm,
                is_batch=self.isBatch
            )
            # run results for single patients
            if not self.isBatch:
                results_single[patient] = patient_experiment.run_experiment()
                self.log.debug("Start processing single results: {}".format(results_single[patient]))
                # super verbose debug information should be logged with debug option
            else:
                # TODO: think about consistent interfaces
                batch_results, self._allFeatureDesp = patient_experiment.run_experiment()
                self.log.info("Start processing batch results with {} batches".format(len(batch_results)))
                batchNo = 0
                for results in batch_results:
                    # here are the result for each feature subset for each patients
                    results_batch[batchNo][patient] = results
                    batchNo += 1
        if self.algorithm == "fm":
            return
        if not self.isBatch:
            results_overall = self.process_results(results_single)
            self.store_results(results_overall, results_single)
            self.log.info("Finished running experiment")
        else:
            self.record_batch_experiment()
            for batchNo, patient_results in results_batch.iteritems():
                # patient_results = results_single
                results_overall = self.process_results(patient_results)
                self.store_results(results_overall, patient_results, self._allFeatureDesp[batchNo])
                self.log.info("Finished running experiment No. {}".format(batchNo))

    def process_results(self, results_single):
        '''
         Processes and aggregates result<s from individual experiments. In addition, a summary is printed to stdout.

        :param results_single:
        :return:
        '''
        self.log.info("Process results")
        # aggregate results
        res_all = list()
        # time of day
        res_night = list()
        res_morning = list()
        res_afternoon = list()
        res_evening = list()
        # time since last meal
        res_meal_first = list()
        res_meal_second = list()
        res_meal_third = list()
        res_meal_fourth = list()
        res_meal_four = list()
        # low and high glucose results
        low_glucose = list()
        low_times = list()
        low_predictions = list()
        high_glucose = list()
        high_times = list()
        high_predictions = list()
        for patient, results in results_single.iteritems():
            # first gather all individual results with predictions
            if results is None: continue
            pres_all = results['performance']['all']
            if pres_all['num_values'] > 0:
                res_all.append(pres_all)
            pres_night = results['performance']['night']
            if pres_night['num_values'] > 0:
                res_night.append(pres_night)
            pres_morning = results['performance']['morning']
            if pres_morning['num_values'] > 0:
                res_morning.append(pres_morning)
            pres_afternoon = results['performance']['afternoon']
            if pres_afternoon['num_values'] > 0:
                res_afternoon.append(pres_afternoon)
            pres_evening = results['performance']['evening']
            if pres_evening['num_values'] > 0:
                res_evening.append(pres_evening)
            pres_meal_first = results['performance']['meal_60m']
            if pres_meal_first['num_values'] > 0:
                res_meal_first.append(pres_meal_first)
            pres_meal_second = results['performance']['meal_61m-120m']
            if pres_meal_second['num_values'] > 0:
                res_meal_second.append(pres_meal_second)
            pres_meal_third = results['performance']['meal_121m-180m']
            if pres_meal_third['num_values'] > 0:
                res_meal_third.append(pres_meal_third)
            pres_meal_fourth = results['performance']['meal_181m-240m']
            if pres_meal_fourth['num_values'] > 0:
                res_meal_fourth.append(pres_meal_fourth)
            pres_meal_four = results['performance']['meal_240m']
            if pres_meal_four['num_values'] > 0:
                res_meal_four.append(pres_meal_four)

            if self.regression:
                # extract low and high glucose values, times, and predictions
                for i, value in enumerate(results['groundtruth']):
                    if value < Constant.HYPOGLYCEMIA_THRESHOLD:
                        # hypoglycemia
                        low_glucose.append(value)
                        low_times.append(results['times'][i])
                        low_predictions.append(results['predictions'][i])
                    if value > Constant.HYPERGLYCEMIA_THRESHOLD:
                        high_glucose.append(value)
                        high_times.append(results['times'][i])
                        high_predictions.append(results['predictions'][i])

        results_overall = dict()

        if self.regression:
            results_overall['all'] = self.aggregate_results(res_all)
            results_overall['night'] = self.aggregate_results(res_night)
            results_overall['morning'] = self.aggregate_results(res_morning)
            results_overall['afternoon'] = self.aggregate_results(res_afternoon)
            results_overall['evening'] = self.aggregate_results(res_evening)
            results_overall['meal_60m'] = self.aggregate_results(res_meal_first)
            results_overall['meal_61m-120m'] = self.aggregate_results(res_meal_second)
            results_overall['meal_121m-180m'] = self.aggregate_results(res_meal_third)
            results_overall['meal_181m-240m'] = self.aggregate_results(res_meal_fourth)
            results_overall['meal_240m'] = self.aggregate_results(res_meal_four)
            results_overall['hypoglycemia'] = compute_performance_metrics(
                groundtruth=low_glucose,
                predictions=low_predictions)
            self.log.debug("low_glucose: {} and prediction: {}".format(low_glucose, low_predictions))
            results_overall['hyperglycemia'] = compute_performance_metrics(
                groundtruth=high_glucose,
                predictions=high_predictions
            )
            self.log.debug("high_glucose: {} and prediction: {}".format(high_glucose, high_predictions))
            self.log.info("Overall Prediction performance: {}".format(results_overall))
        else:
            results_overall['all'] = self.aggregate_clf_results(res_all)
            results_overall['night'] = self.aggregate_clf_results(res_night)
            results_overall['morning'] = self.aggregate_clf_results(res_morning)
            results_overall['afternoon'] = self.aggregate_clf_results(res_afternoon)
            results_overall['evening'] = self.aggregate_clf_results(res_evening)
            results_overall['meal_60m'] = self.aggregate_clf_results(res_meal_first)
            results_overall['meal_61m-120m'] = self.aggregate_clf_results(res_meal_second)
            results_overall['meal_121m-180m'] = self.aggregate_clf_results(res_meal_third)
            results_overall['meal_181m-240m'] = self.aggregate_clf_results(res_meal_fourth)
            results_overall['meal_240m'] = self.aggregate_clf_results(res_meal_four)

        return results_overall

    def aggregate_clf_results(self, results):
        result = dict()
        result['accuracy'] = self.average_dict_items(results, 'accuracy')
        result['precision'] = self.average_dict_items(results, 'precision')
        result['recall'] = self.average_dict_items(results, 'recall')
        result['f1'] = self.average_dict_items(results, 'f1')
        result['num_values'] = self.average_dict_items(results, 'num_values')
        return result

    def aggregate_results(self, results):
        """
        Aggregates (averages) given list of result dictionaries containing MAE, MdAE, RMSE, SMAPE, and num_values
        :param results:
        :return:
        """
        result = dict()
        result['MAE'] = self.average_dict_items(results, 'MAE')
        result['MdAE'] = self.average_dict_items(results, 'MdAE')
        result['RMSE'] = self.average_dict_items(results, 'RMSE')
        result['SMAPE'] = self.average_dict_items(results, 'SMAPE')
        result['num_values'] = self.average_dict_items(results, 'num_values')
        return result

    @staticmethod
    def average_dict_items(listOfDicts, key):
        # filter None values
        filtered = filter(lambda x: x[key], listOfDicts)
        num_items = len(filtered)
        if num_items == 0:
            return None
        return 1.0 * sum([item[key] for item in filtered]) / num_items

    def store_results(self, results_overall, results_single, featureDesp=None):
        """
        Store results in the database.
        :return:
        """
        # store experiment run and retrieve id
        # store aggregate results
        # store patient level results
        # happy end
        self.experiment_id = self.store_experiment(mode='r', featureDesp=featureDesp)
        self.log.debug("Stored experiment with id {}".format(self.experiment_id))
        self.store_aggregate_results(results_overall)
        for patientId in self.patientIDs:
            if results_single[patientId] is None: continue
            self.store_patient_results(results_single[patientId], patientId)
        self.log.info("Finished experiment {}".format(self.experiment_id))

    def store_aggregate_results(self, results):
        for subset in results:
            if self.regression:
                self.store_aggregate_results_subset(results[subset], subset)
            else:
                self.store_aggregate_clf_results_subset(results[subset], subset)

    def store_aggregate_clf_results_subset(self, results, subset):
        self.log.info("Store aggregate results for experiment {} and subset {}".format(self.experiment_id, subset))
        with self.con:
            self.log.info("resulst: {}".format(results))
            cur = self.con.cursor()
            query = "INSERT IGNORE INTO BG_clf_aggregate (experiment_run_id, subset, num_values, accuracy, `precision`, recall, f1)" \
                    "VALUES (%(experiment_run_id)s, %(subset)s,%(num_values)s, %(accuracy)s, %(precision)s, %(recall)s, %(f1)s)"
            self.log.debug("store_aggregate_clf_results_subset query: '" + query + "'")
            cur.execute(query, {
                "experiment_run_id": self.experiment_id,
                "subset": subset,
                "num_values": results['num_values'],
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "f1": results['f1']
            })

    def store_aggregate_results_subset(self, results, subset):
        self.log.info("Store aggregate results for experiment {} and subset {}".format(self.experiment_id, subset))
        with self.con:
            self.log.info("resulst: {}".format(results))
            cur = self.con.cursor()
            query = "INSERT INTO BG_result_aggregate (experiment_run_id, subset, num_values, " \
                    " MAE, MdAE, RMSE, SMAPE) " \
                    " VALUES (%(experiment_run)s, %(subset)s, %(num_values)s, %(MAE)s, %(MdAE)s, %(RMSE)s, %(SMAPE)s)"

            self.log.debug("store_aggregate_results_subset query: '" + query + "'")

            cur.execute(query, {
                "experiment_run": self.experiment_id,
                "num_values": results['num_values'],
                "subset": subset,
                "MAE": results['MAE'],
                "MdAE": results['MdAE'],
                "RMSE": results['RMSE'],
                "SMAPE": results['SMAPE']
            })

    def store_patient_results(self, results, patientId):
        for subset in results['performance']:
            # TODO: just set a default value (e.g. "all") for features in alrorithms/settings where no feature subset
            # is selected per patient -> this avoids consistency/maintainability problems
            if self.algorithm == "rf" or self.algorithm == "et":
                self.store_patient_results_subset(results['performance'][subset], patientId, subset, results['params'],
                                                  results['featureDesp'])
            else:
                self.store_patient_results_subset(results['performance'][subset], patientId, subset, results['params'],
                                                  None)

        self.store_patient_predictions(patientId, results["predictions"], results['indices'], results['groundtruth'])

    def store_patient_results_subset(self, results, patientId, subset, params, features):
        self.log.info("Store patient results for experiment {}, patient {} and subset {}".format(
            self.experiment_id, patientId, subset))
        with self.con:
            cur = self.con.cursor()

            query = "INSERT IGNORE INTO BG_results_patient (experiment_run_id, subset, patientID, num_values, " \
                    " MAE, MdAE, RMSE, SMAPE, params, features) " \
                    " VALUES (%(experiment_run)s, %(subset)s, %(patient_id)s, %(num_values)s, %(MAE)s, %(MdAE)s, %(RMSE)s, %(SMAPE)s,%(params)s, %(features)s)"
            cur.execute(query, {
                "experiment_run": self.experiment_id,
                "patient_id": patientId,
                "num_values": results['num_values'],
                "subset": subset,
                "MAE": results['MAE'],
                "MdAE": results['MdAE'],
                "RMSE": results['RMSE'],
                "SMAPE": results['SMAPE'],
                "params": params,
                "features": features
            })

    def store_patient_predictions(self, patientId, predictions, indices, gt):
        self.log.info("Store patient predictions for patient {} and experiment {}".format(patientId, self.experiment_id))
        with self.con:
            cur = self.con.cursor()
            ## for executemany, the types needed to set to %s
            pred_query = "INSERT IGNORE INTO BG_prediction_run (experiment_run_id, patientID, value, pos, `gt-test`) VALUES (%s, %s, %s, %s, %s)"
            assert (len(predictions) == len(indices))
            assert (len(predictions) == len(gt))
            cur.executemany(pred_query, [
                (int(self.experiment_id), int(patientId), float(predictions[i]), int(indices[i]), float(gt[i])) for i in
                range(len(predictions))])

    def store_experiment(self, mode="r", featureDesp="all"):
        """
        Store experiment run in db and return id
        :return:
        """
        self.log.info("Storing Experiment")

        with self.con:
            cur = self.con.cursor()
            query = "INSERT INTO BG_experiment_run (model, parameters, svn_rev, experiment_id, features, `type`) " \
                    " VALUES (%(model)s, %(parameters)s, %(svn_rev)s, %(experiment_id)s, %(features)s, %(type)s)"
            cur.execute(query, {
                'model': self.algorithm,
                'parameters': "-",
                'svn_rev': self.svn_rev,
                'experiment_id': self.batchId,
                'features': featureDesp,
                'type': mode
            })
            return cur.lastrowid

    def record_batch_experiment(self):
        with self.con:
            cur = self.con.cursor()
            # TODO: add dynamic description of each batch
            query = "INSERT INTO BG_experiment (description) VALUES ('{description}')".format(
                description="All possible group sub feature set")
            cur.execute(query)
            self.batchId = cur.lastrowid


if __name__ == '__main__':
    # command line arguments
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-a', '--algorithm', help='Prediction Algorithm', required=False)
    parser.add_argument('-r', '--revision', help='Source code revision', required=False)
    # parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-m', '--mode', choices=('c', 'r'), required=True)
    args = parser.parse_args()

    instance = PredictionExperiment(algorithm=args.algorithm)
    if args.mode == 'c':
        instance.regression = False
        if args.algorithm == "global":
            instance.run_global_clf_experiment()
        else:
            instance.run_clf_experiment()
    elif args.mode == 'r':
        instance.regression = True
        instance.run_regression_experiments()
