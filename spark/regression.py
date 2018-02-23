from pyspark.ml.regression import (LinearRegression, GeneralizedLinearRegression,
                                   DecisionTreeRegressor, RandomForestRegressor,
                                   GBTRegressor, AFTSurvivalRegression,
                                   IsotonicRegression)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml import Pipeline


def linear_regression(trainingDataFrame, maxIter=10, regParam=0.3, elasticNetParam=0.8):
    lr = LinearRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
    lrModel = lr.fit(trainingDataFrame)
    result = {}
    result["model"] = lrModel
    result["summary"] = lrModel.summary
    result["intercept"] = lrModel.intercept
    result["coefficients"] = lrModel.coefficients
    return result


def generalized_linear_regression(trainingDataFrame, family="gaussian", link="identity",
                                  maxIter=10, regParam=0.3):
    glr = GeneralizedLinearRegression(family=family, link=link, maxIter=maxIter, regParam=regParam)
    glrModel = glr.fit(trainingDataFrame)
    result = {}
    result["model"] = glrModel
    result["summary"] = glrModel.summary
    result["intercept"] = glrModel.intercept
    result["coefficients"] = glrModel.coefficients
    return result


def decision_tree_regression(trainingDataFrame, maxCategories=4):
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                   maxCategories=maxCategories).fit(trainingDataFrame)
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[featureIndexer, dt])
    dtModel = pipeline.fit(trainingDataFrame)
    result = {}
    result["model"] = dtModel
    result["summary"] = dtModel.stages[1]
    return result


def random_forest_regression(trainingDataFrame, maxCategories=4, numTrees=10):
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                   maxCategories=4).fit(trainingDataFrame)
    rf = RandomForestRegressor(featuresCol="indexedFeatures", numTrees=numTrees)
    pipeline = Pipeline(stages=[featureIndexer, rf])
    rfModel = pipeline.fit(trainingDataFrame)
    result = {}
    result["model"] = rfModel
    result["summary"] = rfModel.stages[1]
    return result


def gradient_boosted_tree_regression(trainingDataFrame, maxCategories=4, maxIter=10):
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                   maxCategories=maxCategories).fit(trainingDataFrame)
    gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=maxIter)
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    gbtModel = pipeline.fit(trainingDataFrame)
    result = {}
    result["model"] = gbtModel
    result["summary"] = gbtModel.stages[1]
    return result


def survival_regression(trainingDataFrame, quantileProbabilities=[0.3, 0.6],
                        quantilesCol="quantiles"):
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                                quantilesCol=quantilesCol)
    aftModel = aft.fit(trainingDataFrame)
    result = {}
    result["model"] = aftModel
    result["intercept"] = aftModel.intercept
    result["coefficients"] = aftModel.coefficients
    result["scale"] = aftModel.scale
    return result


def isotonic_regression(trainingDataFrame):
    iso = IsotonicRegression()
    isoModel = iso.fit(trainingDataFrame)
    result = {}
    result["model"] = isoModel
    result["boundaries"] = isoModel.boundaries
    result["predictions"] = isoModel.predictions
    return result


def predict(model, dataFrame):
    predictions = model.transform(dataFrame)
    return predictions


def evaluator(predictions, metricName="rmse"):
    #  "rmse" : root mean squared error - "mse": mean squared error
    #  "r2": R^2^ metric - "mae": mean absolute error
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                    metricName=metricName)
    return evaluator.evaluate(predictions)
