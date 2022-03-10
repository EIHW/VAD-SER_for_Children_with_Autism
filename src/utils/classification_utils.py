import numpy as np
from ordered_set import OrderedSet

def convert_labels_string_to_int(labels_as_string, label_dict_int_to_string=None, one_hot=False):
    """
    Takes a list of string labels and converts them to a list of integers by assigning each class one integer.
    The mapping is saved in the label_dict.
    :param labels_as_string: numpy array with class labels as strings, e.g., ["dog", "cat", "cat", "dog"]
    :return: numpy array of ints, class label dictionary
    """
    if label_dict_int_to_string == None:
        # generate new label_dict
        label_dict_int_to_string = {}
        label_set = OrderedSet(labels_as_string)
        for label_int, label_string in enumerate(label_set):
            label_dict_int_to_string[label_int] = label_string
    if one_hot:
        labels_as_int = np.zeros(labels_as_string.shape + (len(label_dict_int_to_string.keys()),), dtype=int)
    else:
        labels_as_int = np.empty(labels_as_string.shape, dtype=int)
    for label_int in label_dict_int_to_string.keys():
        label_string = label_dict_int_to_string[label_int]
        if one_hot:
            labels_as_int[labels_as_string == label_string, label_int] = 1
        else:
            labels_as_int[labels_as_string == label_string] = label_int
    return labels_as_int, label_dict_int_to_string


