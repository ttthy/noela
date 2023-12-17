import time
import logging
from collections import OrderedDict

totaltime = OrderedDict()
start_at = OrderedDict()


def round_time():
    # return int(round(time.time()*1000))
    return time.time()


def tik(name="total"):
    start_at[name] = round_time()
    return start_at[name]


def tok(name="total"):
    if name not in start_at:
        raise Exception("`{}` not tik yet".format(name))
    if name not in totaltime:
        totaltime[name] = 0
    totaltime[name] += round_time() - start_at[name]
    return totaltime[name]


def get_time(name="total"):
    if name not in start_at:
        raise Exception("not tik yet")
    current_duration = 0
    if name not in totaltime:
        totaltime[name] = round_time() - start_at[name]
    else:
        current_duration = totaltime[name]
    return current_duration


def print_time(name=None):
    if name is not None:
        print("Running {} : {:>10}".format(name, humanize_time(totaltime[name])))
    else:
        print("\n--------- RUNNING TIME --------")
        # if "total" not in totaltime:
        #     tok()
        if "total" in totaltime:
            print("--- {:>15} : {:>10}".format(
                "Total time", humanize_time(totaltime["total"])))
        for name, t in totaltime.items():
            if name != "total":
                print("--- {:>15} : {:>10}".format(name, humanize_time(t)))
        print("-------------------------------")


def print_time_n_remove(name):
    print_time(name)
    start_at.pop(name)
    totaltime.pop(name)


def time_reset():
    global totaltime
    global start_at
    totaltime = OrderedDict()
    start_at = OrderedDict()
    tik()


def humanize_time(second):
    """
    :param second: time in seconds
    :return: human readable time (hours, minutes, seconds)
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


############################### coloring ###########################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
