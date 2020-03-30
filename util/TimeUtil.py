import time


def tohour(duration):
    """
    convert timedelta to hours
    :param duration:
    :return:
    """
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    return hours


def str_to_date_time(t):
    return time.strptime(t, "'%d/%m/%y %H:%M:%S")
