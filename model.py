from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.sql import SparkSession


class Model:
    """
    Model class for wine prediction in PySpark MLLib. Implements logistic
    regression and random forest classification models. It has two methods: 
    train and predict.
    """
    def __init__(self):
        self.spark = (
            SparkSession
            .builder
            .appName("Wine Prediction")
            .config("spark.driver.memory", "30g")
            .config("spark.executor.memory", "30g")
            .config("spark.default.parallelism", "8")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
            .config("spark.driver.maxResultSize", "30g")
            .getOrCreate())
        self.spark.sparkContext.setLogLevel("OFF")
        self.evaluator = MulticlassClassificationEvaluator(
            labelCol="quality",
            predictionCol="prediction",
            metricName="f1")

    def logistic_regression(self, input_df, test_df):
        """
        Implements logistic regression with cross validation.

        :param input_df: Input dataframe
        :type input_df: pyspark.sql.DataFrame
        :param test_df: Evaluation dataframe
        :type test_df: pyspark.sql.DataFrame
        :return: cv_model, test_eval
        :rtype: Tuple[CVModel, int]
        """
        lr = LogisticRegression(
            maxIter=30,
            regParam=0.3,
            elasticNetParam=0.3,
            featuresCol="features",
            labelCol="quality"
        )
        search_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.0, 0.3, 0.6]) \
            .addGrid(lr.elasticNetParam, [0.4, 0.6, 0.8]).build()
        cv = CrossValidator(
            estimator=lr,
            estimatorParamMaps=search_grid,
            evaluator=self.evaluator,
            numFolds=3
        )
        cv_model = cv.fit(input_df)
        train_predictions = cv_model.transform(input_df)
        test_predictions = cv_model.transform(test_df)
        train_eval = self.evaluator.evaluate(train_predictions)
        test_eval = self.evaluator.evaluate(test_predictions)
        print("Logistic regression F1 on train data with CV = %g" % train_eval)
        print("Logistic regression F1 on test data with CV = %g" % test_eval)
        return cv_model, test_eval

    def random_forest(self, input_df, test_df):
        """
        Implements random forest with cross validation.

        :param input_df: Input dataframe
        :type input_df: pyspark.sql.DataFrame
        :param test_df: Evaluation dataframe
        :type test_df: pyspark.sql.DataFrame
        :return: cv_model, test_eval
        :rtype: Tuple[CVModel, int]
        """
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="quality",
            numTrees=100,
            maxBins=128,
            maxDepth=20,
            minInstancesPerNode=5,
            seed=33
        )
        search_grid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100, 200]) \
            .addGrid(rf.maxDepth, [10, 20, 30]).build()
        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=search_grid,
            evaluator=self.evaluator,
            numFolds=3
        )
        cv_model = cv.fit(input_df)
        train_predictions = cv_model.transform(input_df)
        test_predictions = cv_model.transform(test_df)
        train_eval = self.evaluator.evaluate(train_predictions)
        test_eval = self.evaluator.evaluate(test_predictions)
        print("Random Forest F1 on traning data = %g" % train_eval)
        print("Random Forest F1 on eval data = %g" % test_eval)
        return cv_model, test_eval

    def train(self, input_df, test_df, features):
        """
        Train method. Fits logistic regression and random forest, then selects
        the best one according to their evaluation metrics, saves it.

        :param input_df: Input dataframe
        :type input_df: pyspark.sql.DataFrame
        :param test_df: Evaluation dataframe
        :type test_df: pyspark.sql.DataFrame
        :param features: Features list used in training
        :type features: List
        """
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        input_df = assembler.transform(input_df)
        test_df = assembler.transform(test_df)
        lr_model, lr_accuracy = self.logistic_regression(input_df, test_df)
        rf_model, rf_accuracy= self.random_forest(input_df, test_df)
        if lr_accuracy > rf_accuracy:
            model = lr_model
        else:
            model = rf_model
        model.write().overwrite().save("model/prod.model")

    def predict(self, test_df, features):
        """
        Predict method. Loads saved model and applies it to the test data.

        :param test_df: Evaluation dataframe
        :type test_df: pyspark.sql.DataFrame
        :param features: Features list used in prediction
        :type features: List
        """
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        model = CrossValidatorModel.load("model/prod.model")
        test_df = assembler.transform(test_df)
        predictions = model.transform(test_df)
        score = self.evaluator.evaluate(predictions)
        print("F1 score on test data = %g" % score)
