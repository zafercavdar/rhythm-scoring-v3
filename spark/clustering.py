from pyspark.mllib.clustering import KMeans, GaussianMixture, PowerIterationClustering, LDA, \
                                     BisectingKMeans
from math import sqrt
from src.preprocess.transform import setup_spark
from pyspark.rdd import RDD
from pyspark.mllib.linalg import Vectors


def k_means(unclustered_data, number_of_clusters, max_iterations=10, initialization_mode='random',
            initialization_steps=2, initial_model=None, seed=None):
    def error(clusters, point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    if not (initialization_mode == 'k-means||' or initialization_mode == 'random'):
        raise ValueError("Initialization mode for BisectingKMeans can be either \
                          random or k-means||. {} is invalid.".format(initialization_mode))

    if number_of_clusters < 1:
        raise ValueError("While clustering with K-means, \
                the given number of clusters is not positive")

    clusters = KMeans.train(rdd=unclustered_data, k=number_of_clusters,
                            maxIterations=max_iterations, initializationMode=initialization_mode,
                            seed=seed, initializationSteps=initialization_steps,
                            initialModel=initial_model)

    # WSSSE = Within Set Sum of Squared Error
    WSSSE = unclustered_data.map(lambda point: error(clusters, point)).reduce(lambda x, y: x + y)
    return [clusters, WSSSE]


def bisecting_k_means(unclustered_data, number_of_clusters, max_iterations=5,
                      seed=None, min_divisible_cluster_size=1.0):

    if number_of_clusters < 1:
        raise ValueError("While clustering with BisectingKMeans, \
                the given number of clusters is not positive")

    model = BisectingKMeans.train(rdd=unclustered_data,
                                  k=number_of_clusters,
                                  maxIterations=max_iterations,
                                  seed=seed,
                                  minDivisibleClusterSize=min_divisible_cluster_size
                                  )
    cost = model.computeCost(unclustered_data)
    return [model, cost]


def gaussian_mixture(unclustered_data, number_of_clusters, max_iterations=100, seed=None,
                     initial_model=None):

    if number_of_clusters < 1:
        raise ValueError("While clustering with GaussianMixture, \
                the given number of clusters is not positive")

    gmm = GaussianMixture.train(rdd=unclustered_data, k=number_of_clusters,
                                maxIterations=max_iterations, seed=seed, initialModel=initial_model)
    parameters = []
    for i in range(number_of_clusters):
        parameters.append({"weight": gmm.weights[i], "mu": gmm.gaussians[i].mu,
                           "sigma": gmm.gaussians[i].sigma.toArray()})
    return [gmm, parameters]


def power_iteration_clustering(unclustered_data, number_of_clusters, max_iterations=10,
                               init_mode='random'):
    if number_of_clusters < 1:
        raise ValueError("While clustering with PowerIterationClustering, \
                the given number of clusters is not positive")

    model = PowerIterationClustering.train(rdd=unclustered_data, k=number_of_clusters,
                                           maxIterations=max_iterations, initMode=init_mode)
    assignments = model.assignments().collect()
    return [model, assignments]


def latent_dirichlet_allocation(unclustered_data, number_of_clusters, max_iterations=20,
                                doc_concentration=-1.0, topic_concentration=-1.0, seed=None,
                                checkpoint_interval=10, optimizer='em'):

    if number_of_clusters < 1:
        raise ValueError("While clustering with LDA, \
                the given number of clusters is not positive")

    parsedData = unclustered_data.map(lambda lst: Vectors.dense(lst))
    corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
    ldaModel = LDA.train(rdd=corpus, k=number_of_clusters, maxIterations=max_iterations,
                         docConcentration=doc_concentration, topicConcentration=topic_concentration,
                         seed=seed, checkpointInterval=checkpoint_interval, optimizer=optimizer)
    topics = ldaModel.topicsMatrix()
    return [ldaModel, topics]


def predict(saved_model, data_point):
    return saved_model.predict(data_point)


def save_model(model, location):
    sc = setup_spark()
    model.save(sc, location)


def load_model(model_class, location):
    sc = setup_spark()
    return model_class.load(sc, location)


def cluster(method, unclustered_data, number_of_clusters, **kwargs):
    functions = {
        "KMeans": k_means,
        "BisectingKMeans": bisecting_k_means,
        "GaussianMixture": gaussian_mixture,
        "PowerIterationClustering": power_iteration_clustering,
        "LDA": latent_dirichlet_allocation,
    }

    func = functions.get(method, None)
    if func is None:
        raise ValueError("In clustering, no method found named {}".format(func))

    if not isinstance(unclustered_data, RDD):
        sc = setup_spark()
        unclustered_data = sc.parallelize(unclustered_data)

    model = func(unclustered_data, number_of_clusters, **kwargs)

    return {
        'type': 'model',
        'model': model[0],
        'meta': model[1]
    }
