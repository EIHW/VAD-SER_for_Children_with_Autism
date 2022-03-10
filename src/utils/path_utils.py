from os.path import basename, splitext, exists
from os import makedirs, walk
from src.utils.constants import *


def make_directory(dir):
    if not exists(dir):
        makedirs(dir)

def get_subdirecotries(dir, startswith=None):
    if startswith == None:
        return [basename(x[0]) for x in walk(dir)]
    else:
        return [basename(x[0]) for x in walk(dir) if basename(x[0]).startswith(startswith)]

def add_run_directory(result_dir, prefix, run=None):
    subdirs = get_subdirecotries(result_dir, startswith=prefix)
    subdirs.sort()
    if (len(subdirs)) == 0 or run == 1:
        current_run = 1
    elif run==None:
        last_run_dir = subdirs[-1]
        last_run = int(last_run_dir.replace(prefix, ""))
        current_run = last_run + 1
    else:
        current_run = run
    run_dir = prefix + str(current_run).zfill(3)
    make_directory(result_dir + run_dir)
    return run_dir + "/", current_run

def new_run_directory(parent_dir, mode, run=None):
    run_dir = ""
    if mode == QUICK_TEST:
        run_dir = "quick_run_"
    elif mode == TRAIN_AND_EVALUATE or mode == TRAIN:
        run_dir = "run_"
    elif mode == INFERENCE:
        run_dir = "inference_"
    elif mode == APPLY_MODEL:
        run_dir = "apply_model_"
    elif mode == EVALUATE:
        run_dir = "evaluation_"
    elif mode == EVALUATE_INDIVIDUALLY:
        run_dir = "individual_evaluation_"
    return add_run_directory(parent_dir, run_dir, run)

