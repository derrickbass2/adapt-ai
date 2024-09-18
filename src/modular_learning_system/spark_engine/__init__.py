from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession


class SparkEngineUtils:
    @staticmethod
    def convert_to_pandas(df):
        return df.toPandas()

    @staticmethod
    def convert_to_spark(df, spark):
        return spark.createDataFrame(df)

    @staticmethod
    def vectorize_features(df, feature_cols):
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        return assembler.transform(df)

    @staticmethod
    def calculate_silhouette_score(df, prediction_col):
        evaluator = ClusteringEvaluator(predictionCol=prediction_col)
        return evaluator.evaluate(df)


class SparkEngine:
    def __init__(self):
        self.spark_session = SparkSession.builder.getOrCreate()

    def read_csv(self, file_path):
        return self.spark_session.read.csv(file_path, header=True, inferSchema=True)

    @staticmethod
    def preprocess_data(df, feature_cols):
        return df.select(*feature_cols)

    @staticmethod
    def cluster_data(df, num_clusters):
        kmeans = KMeans(k=num_clusters, seed=1)
        model = kmeans.fit(df)
        return model.transform(df)

    @staticmethod
    def write_parquet(df, output_path):
        df.write.parquet(output_path)

    @staticmethod
    def train_linear_regression(df, feature_cols, label_col):
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        lr = LinearRegression(labelCol=label_col)
        model = lr.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    @staticmethod
    def train_decision_tree_classifier(df, feature_cols, label_col):
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        dt = DecisionTreeClassifier(labelCol=label_col)
        model = dt.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    @staticmethod
    def train_random_forest_classifier(df, feature_cols, label_col):
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        rf = RandomForestClassifier(labelCol=label_col)
        model = rf.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    @staticmethod
    def evaluate_model(predictions, label_col):
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
        return evaluator.evaluate(predictions)

    @staticmethod
    def calculate_confusion_matrix(predictions, label_col):
        rdd = predictions.select(label_col, "prediction").rdd.map(tuple)
        metrics = MulticlassMetrics(rdd)
        return metrics.confusionMatrix().toArray()

    def train_model(self, df_preprocessed, label_col):
        pass

    def predict(self, model, df):
        pass
