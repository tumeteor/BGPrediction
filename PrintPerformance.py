__author__ = 'markus'

import configuration as cfg
import logging
from argparse import ArgumentParser
import MySQLdb as mdb
import MySQLdb.cursors
import os, errno

import pprint
pp = pprint.PrettyPrinter(indent=2)

class ResultsPrinter:

    def __init__(self, experimentRunId, mode='r'):
        # configure log
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.DEBUG)
        self.log = logging.getLogger("ResultsPrinter")

        self.mode = mode
        self.experimentRunId = experimentRunId

        # set up database connection
        mysql_data = cfg.data['database']
        self.con = mdb.connect(user=mysql_data['user'], passwd=mysql_data['passwd'],
                               db=mysql_data['db'], charset='utf8', host=mysql_data['host'],
                               cursorclass=MySQLdb.cursors.DictCursor)

        self.patientIDs = list()
        self.resultSubsets = list()
        self.model = None
        self.experimentTimestamp = None
        self.mode = None
        self.folder = 'results'
        self.ensureFolderExists(self.folder)
        self.result_file = "{}/exp_run_{}_result_summary.txt".format(self.folder, self.experimentRunId)
        self.newline = "\n"

    def ensureFolderExists(self, directory):
        """
        Ensure directory exists
        """
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def loadPatients(self):
        """
        Retrieve the patientIDs from database
        """
        self.log.info("Loading Patient data")
        with self.con:
            cur = self.con.cursor()
            query = "SELECT id FROM BG_Patient " \
                    "WHERE id not in (9) "
            self.log.debug("loadPatients query: '" + query + "'")
            cur.execute(query)
            logging.debug("{} Patients returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No patients where returned!")
                return
            for row in rows:
                self.patientIDs.append(row['id'])

    def printResults(self):
        self.writeHeader()
        self.loadPatients()

        if self.mode == 'r':
            self.printRegressionResults()
        elif self.mode == 'c':
               self.printClassificationResults()

    def getExperimentInfo(self):
        self.log.info("Retrieving experiment info")
        # retrieve info from db
        with self.con:
            cur = self.con.cursor()
            query = "SELECT model, timestamp, type FROM BG_experiment_run " \
                    "WHERE id = {}".format(self.experimentRunId)
            cur.execute(query)
            result = cur.fetchall()[0]
            self.model = result['model']
            self.experimentTimestamp = result['timestamp']
            self.mode = result['type']
        # construct description string
        typeString="unknown"
        if self.mode == 'c':
            typeString = 'classification'
        if self.mode == 'r':
            typeString = 'regression'
        experimentInfo = "Experiment {} ({}), type: {}, model: {}".format(
            self.experimentRunId, self.experimentTimestamp, typeString, self.model
        )
        self.log.info("Experiment info: " + experimentInfo)
        return experimentInfo

    def printRegressionResults(self):
        # aggregated performance
        self.printRegressionResultsAggregate()
        # results by patient
        self.writeLine("== PATIENT RESULTS ==")
        self.writeNewline()
        for patient in self.patientIDs:
            self.printRegressionResultsByPatient(patient)
        # patient results by subset
        self.writeLine("== SUBSET RESULTS ==")
        self.writeNewline()
        for subset in self.resultSubsets:
            self.printRegressionResultsPatientsBySubset(subset)

    def printRegressionResultsAggregate(self):
        # also store experiment info
        self.log.info("Loading aggregate results")
        with self.con:
            cur = self.con.cursor()
            query = """
             SELECT ra.subset, ra.num_values, ra.MdAE, ra.RMSE, ra.SMAPE
             FROM BG_result_aggregate ra
             JOIN BG_experiment_run er ON ra.experiment_run_id = er.id
             WHERE ra.experiment_run_id = %(id)s
             ORDER BY FIELD(ra.subset, 'all', 'night', 'morning', 'afternoon', 'evening', 'hypoglycemia', 'hyperglycemia', 'meal_60m', 'meal_61m-120m', 
                 'meal_121m-180m', 'meal_181m-240m', 'meal_240m')
            """""
            cur.execute(query, {"id": self.experimentRunId})
            logging.debug("{} results returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No results where returned!")
                return
            self.log.info("Writing aggregate results")
            # print latex table header
            self.writeLine("== Aggregate (averaged) performance ==")
            self.writeNewline()
            self.writeLine("Result subset & num_values & MdAE & RMSE & SMAPE \\\\")
            for row in rows:
                self.resultSubsets.append(row["subset"])
                if long(row["num_values"]) == 0:
                    # handle subsets without valid results (and hence no performance numbers)
                    self.writeLine("{subset} & {num_values} & - & - & - \\\\".format(
                        subset=row["subset"], num_values=row["num_values"]
                    ))
                    continue

                self.writeLine("{subset} & {num_values} & {MdAE:.2f} & {RMSE:.2f} & {SMAPE:.2f} \\\\".format(
                    subset=row["subset"], num_values=row["num_values"], MdAE=row["MdAE"],
                    RMSE=row["RMSE"], SMAPE=row["SMAPE"]
                ))
            self.writeNewline()

    def printRegressionResultsByPatient(self, patientId):
        """
        Print all regression results for the given patient
        """
        self.log.info("Print regression results for patient {}".format(patientId))
        # Retrieve the results
        with self.con:
            cur = self.con.cursor()
            query = """
             SELECT rp.subset, rp.num_values, rp.MdAE, rp.RMSE, rp.SMAPE
             FROM BG_results_patient rp
             JOIN BG_experiment_run er ON rp.experiment_run_id = er.id
             WHERE rp.experiment_run_id = %(experiment)s AND rp.patientID = %(patient)s 
             ORDER BY FIELD(rp.subset, 'all', 'night', 'morning', 'afternoon', 'evening', 'hypoglycemia', 'hyperglycemia', 'meal_60m', 'meal_61m-120m', 
                 'meal_121m-180m', 'meal_181m-240m', 'meal_240m');
             """""
            cur.execute(query, {"experiment": self.experimentRunId, "patient": patientId})
            logging.debug("{} results returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No results where returned!")
                return
            self.log.info("Writing patient results")
            # print latex table header
            self.writeLine("Patient {}:".format(patientId))
            self.writeNewline()
            self.writeLine("Result subset & num_values & MdAE & RMSE & SMAPE \\\\")
            # write latex table rows
            for row in rows:
                if long(row["num_values"]) == 0:
                    # handle subsets without valid results (and hence no performance numbers)
                    self.writeLine("{subset} & {num_values} & - & - & - \\\\".format(
                        subset=row["subset"], num_values=row["num_values"]
                    ))
                    continue

                self.writeLine("{subset} & {num_values} & {MdAE:.2f} & {RMSE:.2f} & {SMAPE:.2f} \\\\".format(
                    subset=row["subset"], num_values=row["num_values"], MdAE=row["MdAE"],
                    RMSE=row["RMSE"], SMAPE=row["SMAPE"]
                ))
            self.writeNewline()

    def printRegressionResultsPatientsBySubset(self, subset):
        """
        Print patient regression results for the given subset
        """
        self.log.info("Print patient lvl regression results for subset {}".format(subset))
        # Retrieve the results
        with self.con:
            cur = self.con.cursor()
            query = """
             SELECT rp.patientID, rp.num_values, rp.MdAE, rp.RMSE, rp.SMAPE
             FROM BG_results_patient rp
             JOIN BG_experiment_run er ON rp.experiment_run_id = er.id
             WHERE rp.experiment_run_id = %(experiment)s AND rp.subset = %(subset)s
             ORDER BY rp.patientID ASC
             """""
            cur.execute(query, {"experiment": self.experimentRunId, "subset": subset})
            logging.debug("{} results returned".format(cur.rowcount))
            rows = cur.fetchall()
            if not rows:
                self.log.error("No results where returned!")
                return
            self.log.info("Writing subset results")
            # print latex table header
            self.writeLine("Subset {}:".format(subset))
            self.writeNewline()
            self.writeLine("Patient & num_values & MdAE & RMSE & SMAPE \\\\")
            # write latex table rows
            for row in rows:
                if long(row["num_values"]) == 0:
                    # handle subsets without valid results (and hence no performance numbers)
                    self.writeLine("{patient} & {num_values} & - & - & - \\\\".format(
                        patient=row["patientID"], num_values=row["num_values"]
                    ))
                    continue

                self.writeLine("{patient} & {num_values} & {MdAE:.2f} & {RMSE:.2f} & {SMAPE:.2f} \\\\".format(
                    patient=row["patientID"], num_values=row["num_values"], MdAE=row["MdAE"],
                    RMSE=row["RMSE"], SMAPE=row["SMAPE"]
                ))
            self.writeNewline()

    def writeHeader(self):
        """
        Write header info to file. Overwrite file if already exists
        """
        self.log.info("Writing header")
        with open(self.result_file, mode="w") as outfile:
            outfile.write("=== RESULT SUMMARY === \n".format(self.experimentRunId))
            outfile.write(self.getExperimentInfo() + self.newline)
            outfile.write(self.newline)

    def writeString(self, string):
        """
        Write a string to the result file
        """""
        with open(self.result_file, mode="a") as outfile:
                outfile.write(string)

    def writeLine(self, line):
        """
        Append a line to the result file
        """
        self.writeString(line + self.newline)

    def writeNewline(self):
        """
        Write a newline to the resultfile
        """
        self.writeString(self.newline)

    def printClassificationResults(self):
        self.log.warn('Evaluating classification results is not implemented yet.')
        pass

if __name__ == '__main__':
    # command line arguments
    parser = ArgumentParser(description='Retrieve prediction results for given experiment and print them in usable (latex) form.')
    parser.add_argument('-id', '--experiment_run_id', help='Id of the experiment run', required=True)
    #parser.add_argument('-m', '--mode', choices=('c', 'r'), required=True, help='Type of the prediction task')
    # Note: we could retrieve the type of the task from the db also
    args = parser.parse_args()

    instance = ResultsPrinter(experimentRunId=args.experiment_run_id)
    instance.printResults()