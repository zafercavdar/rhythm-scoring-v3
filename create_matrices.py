from io_ops import read
import os
import glob
import numpy as np

non_float_fields = ["per", "ref", "jury"]
similarities = ["timing", "cosine", "dtw"]


def get_all_dicts():
    source_path = "{}/results".format(os.getcwd())
    os.chdir(source_path)
    all_dicts = []
    for inputfile in glob.glob("*.txt"):
        dicts = read(inputfile)
        for dic in dicts:
            for key, value in dic.iteritems():
                if key not in similarities and key not in non_float_fields:
                    dic[key] = 1.0 / (1.0001 + float(value))
                elif key not in non_float_fields:
                    dic[key] = float(value)
                elif key == 'jury':
                    if value == 'pass':
                        dic[key] = 1
                    elif value == 'fail':
                        dic[key] = 0
                    else:
                        raise ValueError("Key jury should be pass or fail.")
        all_dicts = all_dicts + dicts
    return all_dicts


def normalize_values(all_dicts):
    max_dict = {}
    for dic in all_dicts:
        for key, value in dic.iteritems():
            if key not in non_float_fields:
                if max_dict.get(key, False):
                    max_dict[key] = max(max_dict[key], abs(value))
                else:
                    max_dict[key] = abs(value)
    for dic in all_dicts:
        for key, value in dic.iteritems():
            if key not in non_float_fields:
                dic[key] = dic[key] / float(max_dict[key])

    return all_dicts


def create_train_from_all_data(normalized_dicts):
    measures = list(normalized_dicts[0].keys())
    for field in non_float_fields:
        measures.remove(field)
    print("All measures: {}".format(measures))
    inputVector = []
    outputVector = []
    for dic in normalized_dicts:
        sample = []
        for measure in measures:
            sample.append(dic[measure])
        inputVector.append(sample)
        outputVector.append(dic['jury'])
    print("Total vectors: {}".format(len(inputVector)))
    return inputVector, outputVector, measures


def create_train_with_equal_pass_fail(normalized_dicts):
    measures = list(normalized_dicts[0].keys())
    non_float_fields = ["per", "ref", "jury"]
    for field in non_float_fields:
        measures.remove(field)
    print("All measures: {}".format(measures))
    passInputVector = []
    passOutputVector = []
    failInputVector = []
    failOutputVector = []
    for dic in normalized_dicts:
        sample = []
        for measure in measures:
            sample.append(dic[measure])
        if dic['jury'] == 1:
            passInputVector.append(sample)
            passOutputVector.append(dic['jury'])
        elif dic['jury'] == 0:
            failInputVector.append(sample)
            failOutputVector.append(dic['jury'])
        else:
            raise ValueError("Key jury should be pass or fail.")

    print("Total passes: {}".format(len(passInputVector)))
    print("Total fails: {}".format(len(failInputVector)))

    if len(passInputVector) > len(failInputVector):
        indices = np.random.permutation(len(failInputVector))
        passInputVector = list(np.array(passInputVector)[indices])
        passOutputVector = passOutputVector[:len(failInputVector)]
    elif len(passInputVector) < len(failInputVector):
        indices = np.random.permutation(len(passInputVector))
        failInputVector = list(np.array(failInputVector)[indices])
        failOutputVector = failOutputVector[:len(passInputVector)]

    inputVector = passInputVector + failInputVector
    outputVector = passOutputVector + failOutputVector
    return inputVector, outputVector, measures


def get_data(method):
    all_dicts = get_all_dicts()
    normalized_dicts = normalize_values(all_dicts)
    if method == 'all':
        vectors = create_train_from_all_data(normalized_dicts)
    elif method == 'equal':
        vectors = create_train_with_equal_pass_fail(normalized_dicts)
    return vectors
