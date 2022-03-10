# executing PCs
LOCAL = "local"
CLUSTER = "cluster"
LAPTOP = "laptop"

# initial model modes
FROM_SCRATCH = "from_scrath"
PRETRAINED = "pretrained"
CONTINUE = "continue"

# training modes
QUICK_TEST = "quick_test"
TRAIN_AND_EVALUATE = "train_and_evaluate"
TRAIN = "train"
EVALUATE = "evaluate"
EVALUATE_INDIVIDUALLY = "evaluate individually"
INFERENCE = "inference"
APPLY_MODEL = "apply_model"

# NN values
CROSS_ENTROPY = "cross_entropy"
ACCURACY = "accuracy"
RECALL = "recall"
F_ONE_SCORE = "f1_score"
UAR = "UAR"
UAP = "UAP"
UAF_ONE = "unweighted_average_f1_score"

#datasets
#DCASE
DCASE_EIGHTEEN = "dcase_2018"
DCASE_EIGHTEEN_INSIDE = "dcase_2018_inside"
DCASE_EIGHTEEN_OUTSIDE = "dcase_2018_outside"
DCASE_EIGHTEEN_TRANSPORT = "dcase_2018_transport"
DCASE_TWENTYONE = "dcase_2018"
DCASE_TWENTYONE_INSIDE = "dcase_2021_inside"
DCASE_TWENTYONE_OUTSIDE = "dcase_2021_outside"
DCASE_TWENTYONE_TRANSPORT = "dcase_2021_transport"

EMBOA = "EMBOA"
DE_ENIGMA = "DE_ENIGMA"

# EMBOA
# speaker
CHILD = "child"
CHILD_ONLY = "child_only"
ALL = "all"

# task modes
MANY_TO_ONE = "many_to_one"
ONE_TO_ONE = "one_to_one"
BUILD_SEQUENCE = "build_sequence"
MANY_TO_MANY = "many_to_many"
CNN_TO_ONE = "cnn_to_one"


# data loader modes
LOAD_BATCHES = "load_batches"
LOAD_ALL = "load_all"