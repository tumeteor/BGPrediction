import datetime
from util.measures import getTimeBinInt
from util.measures import mean
from util.TimeUtil import tohour
import numpy as np
import logging
from itertools import chain, combinations
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


class FeatureManager:
    """
    Class for feature manager
    Add new features here
    """

    def __init__(self, glucoseData, insulinData, carbData, activityData, patientId):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
        self.log = logging.getLogger("FeatureManager")

        self.horizon_minutes = 60

        self.glucoseData = glucoseData
        self.insulinData = insulinData
        self.carbData = carbData
        self.activityData = activityData
        self.patientId = patientId

        self.n_samples = len(self.glucoseData)

        # insulin decay params
        self.gamma = {"basal": 0.5,
                      "rapid": 0.7,
                      "misch": 0.3}

        self.carbRate = 0.5

        self.featureNames = list()

        '''
        assign features into groups, each feature can be assigned to a (or many) group(s)
        that is declared here
        '''

        FeatureGroup.groups = list()
        self.contextFeatureGroup = FeatureGroup(name="Time")
        self.carb_feature_group = FeatureGroup(name="Carb")
        self.insulin_feature_group = FeatureGroup(name="Insulin")
        self.glucoseFeatureGroup = FeatureGroup(name="Glucose")
        self.activity_feature_group = FeatureGroup(name="Activity")

        self._storedGroupIndices = False

    def to_time_bin(self, t):
        """
        Feature time of day
        :param t:
        :return: time period of day
        """
        # TODO: rename method
        time_bin_feature = Feature("timeBin", getTimeBinInt(t))
        self.contextFeatureGroup.add_feature(time_bin_feature)


    def fixed_model(self, ie, bz, carb, il_case, t):
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
        korrekturregel = [40,60]
        carb_increase_rate = 22 # Bz erhoehung 22mg/dl per carb
        carb_factors = {0:0,1:1.5,2:0.5,3:1.0} # carb_factor varies by period /
                                           # of the day, assumed carb_factor at night is as in the evening
        bzc = None # predicted blood glucose
        print il_case
        if il_case == "basal": #basal
            bzc = bz - (korrekturregel[0] * ie)
            print "predicted bg1: {} and old: {} and insulin: {}".format(bzc,bz, ie)
        elif il_case == "rapid": #prandial or rapid + take at mealtime
            carb_factor = carb_factors[getTimeBinInt(t)]
            # expected insulin
            iee = carb * carb_factor
            korrekturfaktor = 15 # describe how much bg after 1 unit rapid /
                                 # insulin is reduced, varied by time of the day
            if ie > iee:
                bzc = bz + ((carb - (carb / iee) * ie) * korrekturfaktor) # assumed fixed as 40
                print "predicted bg2: {} and old: {}".format(bzc,bz)
            else: # when insulin intake is less than expected
                bzc = bz + ((carb - (carb / iee) * ie) * carb_increase_rate)
                print "predicted bg3: {} and old: {}".format(bzc,bz)

        fixed_model_prediction_feature = Feature("fixedModelPrediction",bzc)
        self.contextFeatureGroup.add_feature(fixed_model_prediction_feature)



    def aggregate_history(self, prev_glucose, cur_time, next_time):
        """
        :param cur_time, next_time:
        :return: average of previous glucose values,
                 rolling average of previous glucose values
        """
        # previous values
        # avg of previous glucose values
        # TODO: rename method
        avg_glucose = mean([p['value'] for p in prev_glucose])
        avg_glucose_feature = Feature("avgGlucose", avg_glucose)
        self.glucoseFeatureGroup.add_feature(avg_glucose_feature)
        # previous glucose values (average of last hour) Note: remove since it is covered by the other method for now
        delta = datetime.timedelta(hours=1)
        prev_glucose = mean([a['value'] for a in prev_glucose if a['value'] and a['time'] >= (cur_time - delta)])
        prev_glucose_feature = Feature("prevGlucose", prev_glucose)
        self.activity_feature_group.add_feature(prev_glucose_feature)

        # TODO: trend?
        # avg of previous glucose values in same time bin
        # features.append(avg([p['value'] for p in prev_glucose if getTimeBinInt(p['time']) == cur_time_bin]))
        # decreasing weight w/ difference in time
        avg_prev = 0.0
        avg_prev_w = 0.0
        for prev in prev_glucose:
            t1 = min(next_time.hour, prev['time'].hour)
            t2 = max(next_time.hour, prev['time'].hour)
            time_diff = min(t2 - t1, t1 + 24 - t2) # <= 12
            #t_w = 1.0 / 12.0 * time_diff
            t_w = np.exp(-time_diff)
            avg_prev += t_w * prev['value']
            avg_prev_w += t_w
        time_weight_avg_glucose = avg_prev / avg_prev_w

        time_weight_avg_glucose_feature = Feature("timeWeightAvgGlucose", time_weight_avg_glucose)
        self.glucoseFeatureGroup.add_feature(time_weight_avg_glucose_feature)

    def insulin_decay_concentration(self, c0, timedelta, type):
        """
        assume that the drug is administered intravenously, so that the concentration of the drug
        in the bloodstream jumps almost immediately to its highest level.
        The concentration of the drug then decays exponentially.
        :param c: concentration at t0
        :param timedelta: time distance at t0 + delta
        :param type: insulin type [basal, rapid, misch]
        :return: concentration at t0 + delta
        """
        ct = c0 * np.exp(-self.gamma[type] * tohour(timedelta))
        if ct < 0.001: ct = 0
        return ct

    def carb_linear_absorption(self, c0, timedelta):
        """
        https://diyps.org/2014/05/29/determining-your-carbohydrate-absorption-rate-diyps-lessons-learned/
        :param c0:
        :param timedelta:
        :return:
        """

        ct = - self.carbRate * tohour(timedelta) + c0
        if ct < 0: ct = 0

        return ct

    def aggregate_recent_history(self, look_back, prev_glucose, prev_insulin, prev_carbs, prev_activity, cur_time):
        """
        add aggregated features for recent history
        :param look_back: window size
        :param prev_insulin:
        :param prev_carbs:
        :param prev_activity:
        :param cur_time:
        :return: 5 * (look_back + 1) features
        """
        features = list()

        for k in range(1, look_back + 1):
            delta = datetime.timedelta(hours=k)
            # sum of activity in last k hours
            sum_ac = sum([a['value'] for a in prev_activity if a['value'] and a['time'] >= (cur_time - delta)])
            sum_ac_feature = Feature("sumActivityLast_{}".format(k), sum_ac)
            self.activity_feature_group.add_feature(sum_ac_feature)
            # sum of insulin in last k hours
            sum_il = sum([i['value'] for i in prev_insulin if i['value'] and i['time'] >= (cur_time - delta)])
            sum_il_feature = Feature("sumInsulinLast_{}".format(k),sum_il)
            self.insulin_feature_group.add_feature(sum_il_feature)
            # sum of carbs in last k hours
            sum_carb = sum([c['value'] for c in prev_carbs if c['value'] and c['time'] >= (cur_time - delta)])
            sum_carb_feature = Feature("sumCarbLast_{}".format(k),sum_carb)
            self.carb_feature_group.add_feature(sum_carb_feature)
            # sum of carbs in last k hours
            #if (k-1)%2 and k <= 3: # having too many features
            #    avgGlucose = mean([c['value'] for c in prev_glucose if c['value'] and c['time'] >= (cur_time - delta)])
            #    avgGlucoseFeature = Feature("avgGlucoseLast_{}".format(k),avgGlucose)
            #    self.carb_feature_group.add_feature(avgGlucoseFeature)

            ##### insulin decay feature ####
            #  [self.log.debug("insulin type {}".format(i['type']))  for i in  prev_insulin if i['value'] and i['time'] >= (cur_time - delta)]

            sum_decay_in = sum([self.insulin_decay_concentration(i['value'], (cur_time - i['time']), i['type']) for i in
                              prev_insulin if
                              i['value'] and i['time'] >= (cur_time - delta)])
            sum_decay_in_feature = Feature("sumInsulinDecayLast_{}".format(k),sum_decay_in)
            self.insulin_feature_group.add_feature(sum_decay_in_feature)

            # absorbed insulin: difference between decay and sum of insulin
            # TODO: comment in when we can speficy to ignore for certain models; rf suffers from too many features
            #absIn = sumIl - sumDecayIn
            #self.featureGroups.get(self.Insulin).add_feature("absorbedInsulinLast_{}".format(k), absIn)

            ### carb absorption ####
            ### high feature importance but reduce prediction performance ####

            sum_decay_carb = sum([self.carb_linear_absorption(c['value'], (cur_time - c['time'])) for c in
                                prev_carbs if
                                c['value'] and c['time'] >= (cur_time - delta)])
            sum_decay_carb_feature = Feature("sumCarbAbsorptionLast_{}".format(k), sum_decay_carb)
            self.carb_feature_group.add_feature(sum_decay_carb_feature)

            # absorbed carbs: difference between sum of eaten carbs and decayed carbs
            # TODO: comment in when we can specify to ignore for certain models; rf suffers from too many features
            #absCarbs = sumCarb - sumDecayCarb
            #self.featureGroups.get(self.Carb).add_feature("absorbedCarbsLast_{}".format(k), absCarbs)

        return features


    @staticmethod
    def feature_exhaustive_best_subset(estimator, X, y, lvl=None):
        """
        Exhaustive search for best feature subset
        X: feature matrix
        lvl: expected size of subset
        :return: all possible subsets
        """
        n_features = X.shape[1]
        subsets = chain.from_iterable(combinations(xrange(n_features), k + 1)
                                      for k in xrange(n_features))

        best_score = -np.inf
        best_subset = None
        for subset in subsets:
            if lvl is not None and lvl < n_features:
                if len(subset) != lvl: continue
            score = cross_val_score(estimator, X[:, subset], y, scoring='neg_mean_absolute_errors').mean()
            if score > best_score:
                best_score, best_subset = score, subset

        return best_subset, best_score

    def feature_group_best_subset(self, estimator, X, y):

        n_groups = len(FeatureGroup.groups)
        subsets = chain.from_iterable(combinations(xrange(n_groups), k + 1)
                                      for k in xrange(n_groups))

        best_score = -np.inf
        best_subset = None
        for subset in subsets:
            self.log.info("Feature group subset: {}".format(subset))
            sub_feature_set = set()
            for groupIdx in subset:
                grp = FeatureGroup.groups[groupIdx]
                sub_feature_set = sub_feature_set.union(grp.get_feature_indices())
            sub_feature_set = list(sub_feature_set)
            score = cross_val_score(estimator, X[:, sub_feature_set], y, scoring='neg_mean_absolute_errors').mean()

            self.log.info("Feature subset indices: {}".format(sub_feature_set))
            if score > best_score:
                best_score, best_subset = score, subset
        return best_subset, best_score


    def feature_group_all_subsets(self, X, lvl=2):

        n_groups = len(FeatureGroup.groups)
        subsets = chain.from_iterable(combinations(xrange(n_groups), k + 1)
                                      for k in xrange(n_groups))
        _all_subsets = {}
        for subset in subsets:
            if lvl is not None and lvl < n_groups:
                if len(subset) != lvl: continue
            self.log.info("Feature group subset: {}".format(subset))
            sub_feature_set = set()
            feature_desp = ""
            for groupIdx in subset:
                grp = FeatureGroup.groups[groupIdx]
                sub_feature_set = sub_feature_set.union(grp.get_feature_indices())
                feature_desp += grp.name + " ({} features),".format(grp.size())
            self.log.info("Feature subset indices: {}".format(sub_feature_set))
            sub_feature_set = list(sub_feature_set)
            _all_subsets[subset] = (X[:,sub_feature_set],feature_desp)
        return _all_subsets


    @staticmethod
    def custom_feature_group(X, group_idx):
        grp = FeatureGroup.groups[group_idx]
        feature_indices = grp.get_feature_indices()
        return X[:,feature_indices]


    @staticmethod
    def get_feature_list():
        features = list()

        for feature in Feature.allFeatureList:
            features.append(feature.value)
        return features


    '''
    Combining features here for each glucose instance
    '''
    def build_feature_matrix(self, look_back):
        """
        Extract features from glucose, insulin, carbs, and activity data.
        :return: Feature matrix with all available features
        """
        self.log.info("Extract features for patient {}".format(self.patientId))

        result = list()
        # generate one instance per glucose value, starting with the second
        labels = list()
        for i in range(1, self.n_samples):
            # time of predicted value
            next_time = self.glucoseData[i]['time']
            # current time
            cur_time = next_time - datetime.timedelta(minutes=self.horizon_minutes)

            # Note 05/11/17: all filtering for predictions with gap less than k hours
            last_time = self.glucoseData[i - 1]['time']

            delta_time = next_time - last_time
            #if delta_time.seconds > 3 * 60 * 60: continue

            labels.append(self.glucoseData[i])


            # observed data
            prev_glucose = [item for item in self.glucoseData[:i] if item['time'] <= cur_time]
            prev_insulin = [item for item in self.insulinData if item['time'] <= cur_time]
            prev_carbs = [item for item in self.carbData if item['time'] <= cur_time]
            prev_activity = [item for item in self.activityData if item['time'] <= cur_time]

            '''
            feature to group addition
            '''
            self.to_time_bin(next_time)
            #self.fixed_model(ie=prev_insulin[-1]['value'], bz=prev_glucose[-1]['value'], carb=prev_carbs[-1]['value'],
                            #il_case=prev_insulin[-1]['type'], t=cur_time)
            self.aggregate_history(prev_glucose, cur_time, next_time)
            self.aggregate_recent_history(look_back, prev_glucose, prev_insulin, prev_carbs, prev_activity, cur_time)

            features = self.get_feature_list()

            result.append(features)

            self.log.debug("number of features: {}".format(len(features)))


        #assert (len(result) == self.n_samples -1) # skip the first sample
        if len(result) > 0:
            self.log.debug("Build feature matrix with {} features.".format(len(result[0])))
            self.log.debug([feature.name for feature in list(set.union(*map(set,[group.group for group in FeatureGroup.groups])))])
        return preprocessing.normalize(np.array(result)), \
               np.array(labels)
               #np.array([row['value'] for row in self.glucose_data[1:]])



class FeatureGroup:
    groups = list()

    def __init__(self, name):
        self.group = set()
        self.name = name
        if not self in FeatureGroup.groups: FeatureGroup.groups.append(self)

    def add_feature(self, feature):
        self.group.add(feature)

    def get_feature_indices(self):
        feature_indices = list()
        for feature in self.group:
            feature_indices.append(feature.get_feature_idx())
        return feature_indices

    def size(self): return len(self.group)
    def empty_group(self): self.group = list()



class Feature:
    allFeatureList = list()

    def __init__(self, name, value):
        self.name = name
        self.value = value
        if self in Feature.allFeatureList:
            # feature objects are compared by name
            # this will remove the object with old value
            Feature.allFeatureList.remove(self)
            # this will append the object with new value
            Feature.allFeatureList.append(self)
        else:
            Feature.allFeatureList.append(self)


    def get_feature_idx(self):
        return self.allFeatureList.index(self)

    def modify_value(self, new_value):
        self.value = new_value

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self):
        return hash(self.name)






