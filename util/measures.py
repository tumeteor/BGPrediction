__author__ = 'markus'
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PlotUtil import save_confusion_matrix
np.seterr(divide='ignore', invalid='ignore')
import pprint
pp = pprint.PrettyPrinter(indent=2)

_night = "night"
_morning = "morning"
_afternoon = "afternoon"
_evening = "evening"

def computePerformanceTimeBinned(groundtruth, predictions, timestamps=None, regression=True, plotConfusionMatrix=False, classes=None, patientId=None,model=None):
    """
    Discretizes timestamps into night, morning, afternoon and evening and returns performance metrics for each time
    bin, as well as overall performance metrics
    :param groundtruth:  list of ground truth values
    :param predictions:  list of prediction vales
    :param timestamps:  list of timestamps
    :return: dictionary of preformance metrics dictionaries for the different time bins
    """
    results = dict()
    # first compute results for all data
    results['all'] = computePerformanceMetrics(groundtruth, predictions) if regression else toClfResult(groundtruth, predictions)
    # discretize timestamps
    timebins = [getTimeBin(timestamp) for timestamp in timestamps]
    # bit ugly :( needs to do for now
    gt_night = list()
    pred_night = list()
    gt_morning = list()
    pred_morning = list()
    gt_afternoon = list()
    pred_afternoon = list()
    gt_evening = list()
    pred_evening = list()
    for idx, time in enumerate(timebins):
        if time == _night:
            gt_night.append(groundtruth[idx])
            pred_night.append(predictions[idx])
        if time == _morning:
            gt_morning.append(groundtruth[idx])
            pred_morning.append(predictions[idx])
        if time == _afternoon:
            gt_afternoon.append(groundtruth[idx])
            pred_afternoon.append(predictions[idx])
        if time == _evening:
            gt_evening.append(groundtruth[idx])
            pred_evening.append(predictions[idx])
    # call measurements function for each time bin
    results[_night] = computePerformanceMetrics(gt_night, pred_night) if regression else toClfResult(gt_night, pred_night)
    results[_morning] = computePerformanceMetrics(gt_morning, pred_morning) if regression else toClfResult(gt_morning, pred_morning)
    results[_afternoon] = computePerformanceMetrics(gt_afternoon, pred_afternoon) if regression else toClfResult(gt_afternoon, pred_afternoon)
    results[_evening] = computePerformanceMetrics(gt_evening, pred_evening) if regression else toClfResult(gt_evening, pred_evening)

    if not regression:
        from sklearn.metrics import confusion_matrix
        np.set_printoptions(precision=2)
        cm = confusion_matrix(groundtruth, predictions)
        cm_night = confusion_matrix(gt_night, pred_night)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_night, classes, patientId, "night", model=model)
        cm_morning = confusion_matrix(gt_morning, pred_morning)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_morning, classes, patientId, "morning", model=model)
        cm_afternoon = confusion_matrix(gt_afternoon, pred_afternoon)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_afternoon, classes, patientId, "afternoon", model=model)
        cm_evening = confusion_matrix(gt_evening, pred_evening)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_evening, classes, patientId, "evening", model=model)

        r = dict()
        r[_night] = dict()
        r[_morning] = dict()
        r[_afternoon] = dict()
        r[_evening] = dict()
        r['all'] = dict()
        i = 0
        for cls in set(groundtruth):
            r['all'][cls] = perClassResult(cm, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_night):
            r[_night][cls] = perClassResult(cm_night, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_morning):
            r[_morning][cls] = perClassResult(cm_morning, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_afternoon):
            r[_afternoon][cls] = perClassResult(cm_afternoon, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_evening):
            r[_evening][cls] = perClassResult(cm_evening, i, classes[cls])
            i += 1
        return results, r
    return results

def computePerformanceMeals(groundtruth, predictions, timestamps=None, carbdata=None, regression=True, plotConfusionMatrix=False, classes=None, patientId=None,model=None):
    """
    Compute performance metrics for subsets of the results based on the time elapsed to the last meal.
    """
    # track time to last meal before measurement
    last_meal_deltas = list()
    for idx, timestamp in enumerate(timestamps):
        last_meal_deltas.append(getMinutesFromLastMeal(carbdata, timestamp))

    # generate subsets based on time elapsed after last meal
    gt_first_hour = list()
    pred_first_hour = list()
    gt_second_hour = list()
    pred_second_hour = list()
    gt_third_hour = list()
    pred_third_hour = list()
    gt_fourth_hour = list()
    pred_fourth_hour = list()
    gt_fourhours = list()
    pred_fourhours = list()
    for i in range(0, len(groundtruth)):
        # compute meal performances
        timedelta = last_meal_deltas[i]
        gt = groundtruth[i]
        pred = predictions[i]
        if timedelta <= 60:
            gt_first_hour.append(gt)
            pred_first_hour.append(pred)
        elif timedelta <= 120:
            gt_second_hour.append(gt)
            pred_second_hour.append(pred)
        elif timedelta <= 180:
            gt_third_hour.append(gt)
            pred_third_hour.append(pred)
        elif timedelta <= 240:
            gt_fourth_hour.append(gt)
            pred_fourth_hour.append(pred)
        if timedelta <= 240:
            gt_fourhours.append(gt)
            pred_fourhours.append(pred)
    # compute performance
    results = dict()
    results["meal_60m"] = computePerformanceMetrics(gt_first_hour, pred_first_hour) if regression else toClfResult(gt_first_hour, pred_first_hour)
    results["meal_61m-120m"] = computePerformanceMetrics(gt_second_hour, pred_second_hour) if regression else toClfResult(gt_second_hour, pred_second_hour)
    results["meal_121m-180m"] = computePerformanceMetrics(gt_third_hour, pred_third_hour) if regression else toClfResult(gt_third_hour, pred_third_hour)
    results["meal_181m-240m"] = computePerformanceMetrics(gt_fourth_hour, pred_fourth_hour) if regression else toClfResult(gt_fourth_hour, pred_fourth_hour)
    results["meal_240m"] = computePerformanceMetrics(gt_fourhours, pred_fourhours) if regression else toClfResult(gt_fourhours, pred_fourhours)

    if not regression:
        from sklearn.metrics import confusion_matrix
        np.set_printoptions(precision=2)
        cm_60m = confusion_matrix(gt_first_hour, pred_first_hour)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_60m, classes, patientId, "meal_60m", model=model)
        cm_61m_120m = confusion_matrix(gt_second_hour, pred_second_hour)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_61m_120m, classes, patientId, "meal_61m-120m", model=model)
        cm_121m_180m = confusion_matrix(gt_third_hour, pred_third_hour)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_121m_180m, classes, patientId, "meal_121m-180m", model=model)
        cm_181m_240m = confusion_matrix(gt_fourth_hour, pred_fourth_hour)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_181m_240m, classes, patientId, "meal_181m-240m", model=model)
        cm_240m = confusion_matrix(gt_fourhours, pred_fourhours)
        if plotConfusionMatrix:
            save_confusion_matrix(cm_240m, classes, patientId, "meal_240m", model=model)

        r = dict()
        r["meal_60m"] = dict()
        r["meal_61m-120m"] = dict()
        r["meal_121m-180m"] = dict()
        r["meal_181m-240m"] = dict()
        r["meal_240m"] = dict()
        i = 0
        for cls in set(gt_first_hour):
            r["meal_60m"][cls] = perClassResult(cm_60m, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_second_hour):
            r["meal_61m-120m"][cls] = perClassResult(cm_61m_120m, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_third_hour):
            r["meal_121m-180m"][cls] = perClassResult(cm_121m_180m, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_fourth_hour):
            r["meal_181m-240m"][cls] = perClassResult(cm_181m_240m, i, classes[cls])
            i += 1
        i = 0
        for cls in set(gt_fourhours):
            r["meal_240m"][cls] = perClassResult(cm_240m, i, classes[cls])
            i += 1

        return results, r
    return results



def getMinutesFromLastMeal(carbdata, timestamp):
    last_time = datetime.datetime.min
    if carbdata[0]["time"] > timestamp:
        return None
    for i in carbdata:
        cur_time = i['time']
        if cur_time > timestamp:
            return (timestamp - last_time).total_seconds() / 60.0
        else:
            last_time = cur_time


def getTimeBin(timestamp):
    """
    Return the time bin for the given datetime input
    :param timestamp: datetime object
    :return: Time bin ('night', 'morning', 'afternoon', 'evening'
    """
    assert timestamp
    start_night = datetime.time(0, 0, 0)
    end_night = datetime.time(5, 59, 59)
    start_morning = datetime.time(6, 0, 0)
    end_morning = datetime.time(11, 59, 59)
    start_afternoon = datetime.time(12, 0, 0)
    end_afternoon = datetime.time(17, 59, 59)
    start_evening = datetime.time(18, 0, 0)
    end_evening = datetime.time(23, 59, 59)
    if checkTimeInterval(timestamp, start_night, end_night):
        return _night
    if checkTimeInterval(timestamp, start_morning, end_morning):
        return _morning
    if checkTimeInterval(timestamp, start_afternoon, end_afternoon):
        return _afternoon
    if checkTimeInterval(timestamp, start_evening, end_evening):
        return _evening
    raise ArithmeticError

def getTimeBinInt(timestamp):
    binString = getTimeBin(timestamp)
    if binString == _night:
        return 0
    if binString == _morning:
        return 1
    if binString == _afternoon:
        return 2
    if binString == _evening:
        return 3

def checkTimeInterval(datetimeInput, start, end):
    """
    Check whether given datetime input falls into the interval defined by datetime.time values start and end
    :return: boolean
    """
    assert(datetimeInput)
    # TODO: avoid this! use timestamps in the database instead or transform the timestamps to UTC in import!!
    datetimeInput = datetimeInput + datetime.timedelta(minutes=60)
    assert(start < end)
    timeInput = datetime.time(
        datetimeInput.hour,
        datetimeInput.minute,
        datetimeInput.second)
    return start <= timeInput <= end

def computePerformanceMetrics(groundtruth, predictions):
    """
    Given ground truth and prediction values, compute performance measurements and statistics
    :param groundtruth: list of numerical ground truth values
    :param predictions: list of numerical prediction values
    :return: dictionary of performance metrics and statistics
    """
    assert(len(groundtruth) == len(predictions))
    results = dict()
    results['num_values'] = len(predictions)
    if len(predictions) == 0: # in case there were no values
        results['RMSE'] = None
        results['SMAPE'] = None
        results['MAE'] = None
        results['MdAE'] = None
        return results

    results['RMSE'] = mean_squared_error(groundtruth, predictions)
    results['SMAPE'] = mean_absolute_percentage_error(groundtruth, predictions)
    results['MAE'] = mean_absolute_error(groundtruth, predictions)
    results['MdAE'] = median_absolute_error(groundtruth, predictions)
    return results

def mean_absolute_percentage_error(groundtruth, predictions):
    """
    Compute symmertric mean absolute percentage error (SMAPE)
    :param groundtruth:
    :param predictions:
    :return: SMAPE
    """
    gt = np.array(groundtruth)
    pred = np.array(predictions)
    errors = np.abs(pred - gt)
    averages = (np.abs(gt) + np.abs(pred))/2.0
    return 100.0 * np.mean(errors / averages)

def median_absolute_error(groundtruth, predictions):
    """
    Compute median absolute error (MdAE)
    :param groundtruth:
    :param predictions:
    :return: MdAE
    """
    gt = np.array(groundtruth)
    pred = np.array(predictions)
    errors = np.abs(pred - gt)
    return np.median(errors)

def toClfResult(test_y, predictions):
    results = dict()
    results["num_values"] = len(test_y)
    if len(test_y) == 0:
        results["accuracy"] = None
        results["precision"] = None
        results["recall"] = None
        results["f1"] = None
        return results
    results["accuracy"] = accuracy_score(test_y, predictions, normalize=True)
    results["precision"] = precision_score(test_y, predictions,average='weighted')
    results["recall"] = recall_score(test_y, predictions, average='weighted')
    results["f1"] = f1_score(test_y, predictions, average='weighted')

    return results

def perClassResult(cm, i, cls):
    '''

    :param cm: confusion matrix
    :param i: class_i
    :param cls: class label
    :return:
    '''
    r = dict()
    r['tp'] = cm[i,i]
    r['fp'] = np.sum(cm, axis=0)[i] - cm[i, i]  #The corresponding column for class_i - TP
    r['fn'] = np.sum(cm, axis=1)[i] - cm[i, i] # The corresponding row for class_i - TP
    r['tn'] = np.sum(cm) - r['tp'] -r['fp'] -r['fn']
    r['className'] = cls
    r['precision'] = r['tp'] / float(r['tp'] + r['fp'])
    r['recall'] = r['tp'] / float(r['tp'] + r['fn'])
    r['f1'] = 2 * r['precision'] * r['recall'] / (r['precision'] + r['recall'])
    r['noInstances'] = np.sum(cm, axis=1)[i]
    r['accuracy'] = (r['tp'] + r['tn'])/ float(r['tp'] + r['tn'] + r['fp'] + r['fn'])
    return r


MMOL_TO_MG = 18.0182

def convert_mg_to_mmol(x):
    """Convert mg/dl to mmol/l equivalent"""
    x= format( x/MMOL_TO_MG, '.1f')
    return float(x)

def convert_mmol_to_mg(x):
    """Convert mmol/l to mg/dl equivalent"""
    return int(x*MMOL_TO_MG)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

