from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from create_matrices import get_data
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import operator
from time import time

methods_dict = {
    "NaiveBayes": {
        "GaussianNB": GaussianNB,
        # "MultinomialNB": MultinomialNB,
        "BernoulliNB": BernoulliNB
    },
    "DecisionTrees": {
        "DTRegressor": tree.DecisionTreeRegressor,
        "DTClassifier": tree.DecisionTreeClassifier
    },
    "NeuralNetworks": {
        "MLPClassifier":  MLPClassifier,
        "MLPRegressor": MLPRegressor
    },
    "GaussianProcess": {

    },
    "NearestNeighbors": {
        "KNeighborsClassifier": KNeighborsClassifier
    },
    "LinearModels": {
        "SGDClassifier": SGDClassifier,
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression
    },
    "SupportVectorMachines": {
        "SVC": svm.SVC,
        # "NuSVC": svm.NuSVC,
        "LinearSVC": svm.LinearSVC
    },
    "Ensemble": {
        "AdaBoost": AdaBoostClassifier,
        "RandomForest": RandomForestClassifier
    },
    "Discriminant Analysis": {
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis
    }
}


def _accuracy(predicted_output, test_output, boolean=False):
    if not boolean:
        predicted_output = map(lambda x: 1 if x >= 0.5 else 0, predicted_output)
    return accuracy_score(predicted_output, test_output, normalize=True)


def _classify(train_input, train_output, test_input, test_output, **kwargs):
    results = {}
    for category, value in methods_dict.items():
        for name, method in value.iteritems():
            print("Calculating {}".format(name))
            clf = method(**kwargs)
            clf.fit(train_input, train_output)
            predicted = clf.predict(test_input)
            score = _accuracy(predicted, test_output)
            results[str(name)] = score
    print(results)
    return results


def _classify_knn(train_input, train_output, test_input, test_output):
    max_accuracy = 0
    for k in range(1, 100):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_input, train_output)
        predicted = clf.predict(test_input)
        score = _accuracy(predicted, test_output)
        if score > max_accuracy:
            max_accuracy = score
            print("k:{} => accuracy: {}".format(k, score))


def _classify_spec(train_input, train_output, test_input, test_output):
    clf = LinearRegression()
    clf.fit(train_input, train_output)
    predicted = clf.predict(test_input)
    score = _accuracy(predicted, test_output)
    print("Score: {}".format(score))
    # print("MLP Layers: {}".format(clf.n_layers_))
    return clf.coef_
    # print("Coefs: {}".format(clf.coef_))


def _prepare_data(inputVector, outputVector):
    indices = np.random.permutation(len(inputVector))
    test_size = int(round(len(inputVector) * 0.2)) * -1
    inputVector = np.array(inputVector)
    outputVector = np.array(outputVector)
    train_input = inputVector[indices[:test_size]]
    train_output = outputVector[indices[:test_size]]
    test_input = inputVector[indices[test_size:]]
    test_output = outputVector[indices[test_size:]]
    return train_input, train_output, test_input, test_output


def plot_linear_coefs():
    inputVector, outputVector, measures = get_data('all')
    train_input, train_output, test_input, test_output = _prepare_data(inputVector, outputVector)
    # _classify(train_input, train_output, test_input, test_output)
    coef = _classify_spec(train_input, train_output, test_input, test_output)
    zipped = zip(measures, coef)
    print(zipped)
    coef = list(map(lambda x: abs(x), coef))
    plt.pie(coef, labels=measures, autopct='%1.1f%%')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_method_accuracies():
    start = time()
    inputVector, outputVector, measures = get_data('equal')
    train_input, train_output, test_input, test_output = _prepare_data(inputVector, outputVector)
    results = _classify(train_input, train_output, test_input, test_output)
    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    methods = []
    accuracies = []
    numbers = []
    count = 1
    for method, accuracy in sorted_results:
        methods.append(method)
        accuracies.append(accuracy)
        numbers.append(count)
        count += 1
    print(sorted_results)
    min_accuracy = min(accuracies)
    accuracies = list(map(lambda x: abs(x - min_accuracy), accuracies))
    finish = time()
    print("Total time: {}".format(finish-start))
    plt.bar(numbers, accuracies, align='center')
    plt.xticks(numbers, methods, rotation=40)
    plt.show()


plot_linear_coefs()
