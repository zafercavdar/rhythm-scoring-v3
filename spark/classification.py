from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier,
                                       RandomForestClassifier, GBTClassifier,
                                       MultilayerPerceptronClassifier, LinearSVC,
                                       OneVsRest, NaiveBayes)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, IndexToString
from pyspark.ml import Pipeline


def binomial_logistic_regression(trainingDataFrame, maxIter=100, regParam=0.0,
                                 elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                                 standardization=True, aggregationDepth=2):
    lr = LogisticRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam,
                            tol=tol, fitIntercept=fitIntercept, standardization=standardization,
                            aggregationDepth=aggregationDepth)
    lrModel = lr.fit(trainingDataFrame)
    result = {}
    result["model"] = lrModel
    result["summary"] = lrModel.summary  # https://goo.gl/i5UFA6
    result["intercept"] = lrModel.intercept
    result["coefficients"] = lrModel.coefficients
    return result


def multinomial_logistic_regression(trainingDataFrame, maxIter=100, regParam=0.0,
                                    elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                                    threshold=0.5, thresholds=None, standardization=True,
                                    aggregationDepth=2):
    lr = LogisticRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam,
                            tol=tol, fitIntercept=fitIntercept, standardization=standardization,
                            aggregationDepth=aggregationDepth, family="multinomial")
    lrModel = lr.fit(trainingDataFrame)
    result = {}
    result["model"] = lrModel
    result["interceptVector"] = lrModel.interceptVector
    result["coefficientMatrix"] = lrModel.coefficientMatrix
    return result


def decision_tree_classifier(trainingDataFrame, maxCategories=4, maxDepth=5, maxBins=32,
                             minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                             cacheNodeIds=False, checkpointInterval=10, impurity="gini", seed=None):
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel"). \
                   setHandleInvalid("keep").fit(trainingDataFrame)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                   maxCategories=maxCategories).fit(trainingDataFrame)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                                maxDepth=maxDepth, maxBins=maxBins,
                                minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                                maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds,
                                checkpointInterval=checkpointInterval, impurity=impurity, seed=seed)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    dtModel = pipeline.fit(trainingDataFrame)
    result = {}
    result["model"] = dtModel
    result["summary"] = dtModel.stages[2]
    return result


def random_forest_classifier(trainingDataFrame, maxCategories=4, maxDepth=5, maxBins=32,
                             minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                             cacheNodeIds=False, checkpointInterval=10, impurity="gini",
                             numTrees=20, featureSubsetStrategy="auto", seed=None,
                             subsamplingRate=1.0):
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel"). \
                   setHandleInvalid("keep").fit(trainingDataFrame)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                   maxCategories=maxCategories).fit(trainingDataFrame)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                                maxDepth=maxDepth, maxBins=maxBins,
                                minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                                maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds,
                                checkpointInterval=checkpointInterval, impurity=impurity,
                                numTrees=numTrees, featureSubsetStrategy=featureSubsetStrategy,
                                seed=seed, subsamplingRate=subsamplingRate)
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    rfModel = pipeline.fit(trainingDataFrame)
    result = {}
    result["model"] = rfModel
    result["summary"] = rfModel.stages[2]
    return result


def gradient_boosted_tree_classifier(trainingDataFrame, maxCategories=4, maxDepth=5, maxBins=32,
                                     minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                                     cacheNodeIds=False, checkpointInterval=10, lossType="logistic",
                                     maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0):
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel"). \
                   setHandleInvalid("keep").fit(trainingDataFrame)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                   maxCategories=maxCategories).fit(trainingDataFrame)
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                        maxDepth=maxDepth, maxBins=maxBins,
                        minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                        maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds,
                        checkpointInterval=checkpointInterval, lossType=lossType,
                        maxIter=maxIter, stepSize=stepSize, seed=seed,
                        subsamplingRate=subsamplingRate)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])
    gbtModel = pipeline.fit(trainingDataFrame)
    result = {}
    result["model"] = gbtModel
    result["summary"] = gbtModel.stages[2]
    return result


def multilayer_perceptron_classifier(trainingDataFrame, maxIter=100, tol=1e-6, seed=None,
                                     layers=None, blockSize=128, stepSize=0.03, solver="gd",
                                     initialWeights=None):
    mlp = MultilayerPerceptronClassifier(maxIter=maxIter, tol=tol, seed=seed, layers=layers,
                                         blockSize=blockSize, stepSize=stepSize, solver=solver,
                                         initialWeights=initialWeights)
    mlpModel = mlp.fit(trainingDataFrame)
    result = {}
    result["model"] = mlpModel
    return result


def linear_support_vector_machine(trainingDataFrame, maxIter=100, regParam=0.0, tol=1e-6,
                                  fitIntercept=True, standardization=True, threshold=0.0,
                                  aggregationDepth=2):
    lsvc = LinearSVC(maxIter=maxIter, regParam=regParam, tol=tol, fitIntercept=fitIntercept,
                     standardization=standardization, threshold=threshold,
                     aggregationDepth=aggregationDepth)
    lsvcModel = lsvc.fit(trainingDataFrame)
    result = {}
    result["model"] = lsvcModel
    result["coefficients"] = lsvcModel.coefficients
    result["intercept"] = lsvcModel.intercept
    return result


def one_vs_rest_classifier(trainingDataFrame, classifier=None):
    if not classifier:
        classifier = LogisticRegression(regParam=0.01)
    ovr = OneVsRest(classifier=classifier)
    ovrModel = ovr.fit(trainingDataFrame)
    result = {}
    result["model"] = ovrModel
    return result


def naive_bayes(trainingDataFrame, smoothing=1.0, modelType="multinomial", weightCol="weight"):
    nb = NaiveBayes(smoothing=smoothing, modelType=modelType, weightCol=weightCol)
    nbModel = nb.fit(trainingDataFrame)
    result = {}
    result["model"] = nbModel
    return result


def predict(model, dataFrame):
    predictions = model.transform(dataFrame)
    return predictions


def binary_evaluator(predictions, metricName="areaUnderROC"):
    #  "areaUnderROC", "areaUnderPR"
    evaluator = BinaryClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName=metricName)
    return evaluator.evaluate(predictions)


def multiclass_evaluator(predictions, metricName="accuracy"):
    #  "f1", "weightedPrecision", "weightedRecall", "accuracy"
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName=metricName)
    return evaluator.evaluate(predictions)
