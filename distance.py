from scipy.spatial import distance
from numpy import zeros, inf
import numpy as n


def euclidean(u, v):
    return distance.euclidean(u, v)


def canberra(u, v):
    return distance.canberra(u, v)


def braycurtis(u, v):
    return distance.braycurtis(u, v)


def manhattan(u, v):
    return distance.cityblock(u, v)


def chessboard(u, v):
    return distance.chebyshev(u, v)


def cosine(u, v):
    return distance.cosine(u, v)


def correlation(u, v):
    return distance.correlation(u, v)


def percival(u, v):
    def _aligner(ref, perf):
        meandiff = int(n.average(ref)) - int(n.average(perf))
        return perf + meandiff

    def _find_closest(e, list):
        return min(list, key=lambda x: abs(x-e))

    s_rate = 512
    ref = n.int32(n.array(u) * s_rate)
    perf = n.int32(n.array(v) * s_rate)
    Km = 100
    Ks = 0.45
    s = 0.0
    perf = _aligner(ref, perf)
    for x in ref:
        e = float(abs(_find_closest(x, perf) - x))
        e = e**2 / len(ref)
        e = min(e, Km)
        s += e
    for x in perf:
        e = float(abs(_find_closest(x, ref) - x))
        e = e**2 / len(perf)
        e = min(e, Km)
        s += e
    return 100 - (Ks * s)


def timing_similarity(u, v, bpm):
    min_len = min(len(u), len(v))
    difflist = []
    for i in range(0, min_len):
        difflist.append(abs(u[i] - v[i]))
    diffrange = 60. / bpm
    scorelist = []
    if len(difflist) <= 2:
        score = 0
    else:
        for diff in difflist:
            if diff < diffrange / 32.:
                scorelist.append(1)
            elif diff < diffrange / 16.:
                scorelist.append(0.9)
            elif diff < diffrange / 8.:
                scorelist.append(0.8)
            elif diff < diffrange / 4.:
                scorelist.append(0.3)
            elif diff < diffrange / 2.:
                scorelist.append(0.2)
            else:
                scorelist.append(0.1)
        score = float(sum(scorelist[1:-1]) / (len(scorelist) - 2))
    return score


def dtw(x, y):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    """
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            D1[i, j] = manhattan(x[i], y[j])
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    return D1[-1, -1] / sum(D1.shape)


distance_measures = {
    "euclidean": euclidean,
    "canberra": canberra,
    "braycurtis": braycurtis,
    "manhattan": manhattan,
    "chessboard": chessboard,
    "cosine": cosine,
    "correlation": correlation,
    "dtw": dtw,
    "percival": percival,
    # "timing_similarity": timing_similarity
}


def calc_distance(u, v):
    results = {}
    for method, func in distance_measures.iteritems():
        if method == "timing_similarity":
            distance = func(u, v)
        else:
            distance = func(u, v)
        results[str(method)] = distance
    return results
