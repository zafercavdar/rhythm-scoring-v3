from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from create_matrices import get_data


def _shift_normalize_revert(lst):
    lst_mini = min(lst)
    shifted = [x - lst_mini for x in lst]
    shifted_max = max(shifted)
    normalized = [float(x) / shifted_max for x in shifted]
    return [1 - x for x in normalized]


def detect_outliers(data, method):
    """
    Usage of this method:
        detect_outliers(data, "One-Class SVM")
    """
    outliers_fraction = 0.01
    n = len(data)
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                         kernel="rbf", gamma=0.1),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(max_samples=n,
                                            contamination=outliers_fraction)}

    clf = classifiers.get(method, None)
    if clf is None:
        raise ValueError("While detecting outliers, no method found with name {}".format(method))

    clf.fit(data)
    scores_pred = clf.decision_function(data)
    # y_pred = clf.predict(data)
    pred_normalized = _shift_normalize_revert(scores_pred)
    result = []
    for i in range(n):
        result.append((data[i], pred_normalized[i]))
    result.sort(key=lambda x: -x[1])
    return result


data = get_data('all')
size = len(data[0])
list_input = list(data[0])
for x in range(0, size):
    list_input[x] = list(list_input[x])
    list_input[x].append(data[1][x])

result = detect_outliers(list_input, 'One-Class SVM')
count = 0
for data in result:
    if data[1][0] >= 0.3:
        count += 1
        print("{}: {}".format(count, data[1][0]))
