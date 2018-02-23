from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import LassoCV
from create_matrices import get_data


def select_k_best(X, y, k):
    print("Using SelectKBest")
    print("Before: {}".format(X[0]))
    X_new = SelectKBest(f_regression, k=k).fit_transform(X, y)
    print("After: {}".format(X_new[0]))


def variance_threshold(X, y, threshold):
    print("Using VarianceThreshold")
    print("Before: {}".format(X[0]))
    sel = VarianceThreshold(threshold=threshold)
    X_new = sel.fit_transform(X, y)
    print("After: {}".format(X_new[0]))
    indices = [X[0].index(k) for k in X_new[0]]
    print([measures[i] for i in indices])


def select_from_model(X, y):
    print("Using Select From Model")
    print("Before: {}".format(X[0]))
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(X, y)
    n_features = sfm.transform(X).shape[1]
    while n_features > 1:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]
    feature1 = X_transform[:, 0][0]
    ind1 = X[0].index(feature1)
    # feature2 = X_transform[:, 1][0]
    # ind2 = X[0].index(feature2)
    print(measures[ind1])


data = get_data('all')
measures = data[2]
select_from_model(data[0], data[1])
