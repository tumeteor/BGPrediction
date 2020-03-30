from prediction.base_regressor import BaseRegressor
from util.measures import getTimeBinInt, mean_absolute_percentage_error, mean_squared_error, median_absolute_error, convert_mg_to_mmol
import logging
import numpy as np
class FixedModel(BaseRegressor):

    def __init__(self, patientId, dbConnection):
        super(FixedModel, self).__init__(patientId, dbConnection)
        self.horizon_minutes = 60
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
        self.log = logging.getLogger("Fixed model")

        self.korrekturfaktors = [3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 55, 60] # describe how much bg after 1 unit rapid /
                                 # insulin is reduced, varied by time of the day

        self.korrekturregels = [10,20,30,40,50,60,70,80]

    def save_params(self):
        params = "split_ratio: " + str(self.split_ratio)
        return params

    @staticmethod
    def fixed_model(ie, bz, carb, il_case, t, korrekturfaktor=10, korrekturregel=40):
        '''
        predict bloodglucose using fixed model in 3.5 hours?
        :param ie: ingested insulin
        :param bz: current bloodglucose
        :param carb: carb intake
        :param il_case: insulin case 1 (basal) or 2 (prandial)
               TODO: mixed??
        :param t: timestamp
        :return: predicted bz  - bzc
        '''
        carb_increase_rate = 22 # Bz erhoehung 22mg/dl per carb
        carb_factors = {0:0,1:1.5,2:0.5,3:1.0} # carb_factor varies by period /
                                           # of the day, assumed carb_factor at night is as in the evening
        bzc = None # predicted blood glucose
        #print il_case
        if il_case == "rapid": #basal TODO: change to case I and II
            bzc = bz - (korrekturregel * ie)
            #print "predicted bg1: {} and old: {} and insulin: {}".format(bzc,bz, ie)
        elif il_case == "rapid": #prandial or rapid + take at mealtime
            carb_factor = carb_factors[getTimeBinInt(t)]
            # expected insulin
            iee = carb * carb_factor

            if ie > iee:
                bzc = bz + ((carb - (carb / iee) * ie) * korrekturfaktor) # assumed fixed as 40
                #print "predicted bg2: {} and old: {}".format(bzc,bz)
            else: # when insulin intake is less than expected
                bzc = bz + ((carb - (carb / iee) * ie) * carb_increase_rate)
                #print "predicted bg3: {} and old: {}".format(bzc,bz)

        return bzc

    @staticmethod
    def checkperiod(next_time, cur_time):
        delta_time = next_time - cur_time
        return delta_time.seconds >= 2 * 60 * 60 and delta_time.seconds <= 4 * 60 * 60

    def predict(self):

        bestScore = np.inf
        bestParam = None
        g = None
        p = None
        for korrekturfaktor in self.korrekturfaktors:
            groundtruths, predictions = self.build_mealtime_data(korrekturregel=korrekturfaktor)
            if len(predictions) == 0:
                print "number of predictions: {} and number of instances: {}".format(len(predictions),
                                                                                     len(self.glucose_data))
                print "Percentage of predictions: {:0.2f}".format(len(predictions) / float(len(self.glucose_data)))
                return

            rmse = mean_squared_error(groundtruths, predictions)
            if bestScore > rmse:
                bestScore = rmse
                bestParam = korrekturfaktor
                g = groundtruths
                p = predictions

        avg_gt = sum(g) / float(len(g))
        a = [avg_gt] * len(g)
        print "RMSE: {} for fixed model and avg: {} for patient {}".format(bestScore, mean_squared_error(a,p), self.patient_id)
        print "SMAPE: {} for fixed model and avg: {} for patient {}".format(mean_absolute_percentage_error(g,p), mean_absolute_percentage_error(a,p), self.patient_id)
        print "MdSE: {} for fixed model and avg: {} for patient {}".format(median_absolute_error(g,p), median_absolute_error(a,p), self.patient_id)
        print "number of predictions: {} and number of instances: {}".format(len(predictions), len(self.glucose_data))
        print "Percentage of predictions: {:0.2f}".format(len(predictions) / float(len(self.glucose_data)) * 100)




    def build_mealtime_data(self, korrekturfaktor=10, korrekturregel=40):

        n_samples = len(self.glucose_data)

        groundtruths = list()
        predictions = list()

        nBz = 0

        for i in range(1, n_samples):
            # time of predicted value
            next_glucose =  self.glucose_data[i]
            next_time = next_glucose['time']

            cur_time = self.glucose_data[i - 1]['time']
            cur_glucose = self.glucose_data[i - 1]['value']

            delta_time = next_time - cur_time

            if not self.checkperiod(next_time, cur_time):
                continue


            prev_insulins = [item for item in self.insulin_data if item['time'] <= cur_time]
            prev_carbs = [item for item in self.carb_data if item['time'] <= cur_time]

            if len(prev_insulins) == 0 or len(prev_carbs) == 0: continue

            cur_insulin = prev_insulins[-1]
            cur_carb = prev_carbs[-1]

            # if not self.checkperiod(next_time, cur_carb['time']):
            #     self.log.debug("no carb intake at mealtime?")
            #     continue
            #
            if not self.checkperiod(next_time, cur_insulin['time']):
                self.log.debug("no insulin intake at mealtime?")
                continue

            if self.checkperiod(next_time, cur_carb['time']):
                continue

            nBz += 1
            if cur_insulin['type'] != 'rapid': continue

            predictedBz = self.fixed_model(ie=cur_insulin['value'], bz=cur_glucose, carb=cur_carb['value'],
                                           il_case=cur_insulin['type'], t=cur_time, korrekturfaktor=korrekturfaktor, korrekturregel=korrekturregel)

            groundtruths.append(convert_mg_to_mmol(next_glucose['value']))
            predictions.append(convert_mg_to_mmol(predictedBz))
        print "number of consecutive Bz measurements for fixed model: {}".format(nBz)
        assert len(groundtruths) == len(predictions)
        return groundtruths, predictions
















