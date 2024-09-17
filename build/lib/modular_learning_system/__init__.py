from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, CalinskiHarabaszEvaluator, DavisBouldinEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

__all__ = [
    'SparkEngine',
    'SparkEngineUtils'
]


class SparkEngineUtils:
    @staticmethod
    def convert_to_pandas(df):
        """Converts a Spark DataFrame to a Pandas DataFrame."""
        return df.toPandas()

    @staticmethod
    def convert_to_spark(df, spark):
        """Converts a Pandas DataFrame to a Spark DataFrame."""
        return spark.createDataFrame(df)

    @staticmethod
    def vectorize_features(df, feature_cols):
        """Assembles features into a single feature vector column."""
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        return assembler.transform(df)

    @staticmethod
    def calculate_silhouette_score(df, prediction_col):
        """Calculates the Silhouette Score for clustering."""
        evaluator = ClusteringEvaluator(predictionCol=prediction_col)
        return evaluator.evaluate(df)

    @staticmethod
    def calculate_calinski_harabasz_score(df):
        """Calculates the Calinski-Harabasz Score for clustering."""
        evaluator = CalinskiHarabaszEvaluator()
        return evaluator.evaluate(df)

    @staticmethod
    def calculate_davis_bouldin_score(df):
        """Calculates the Davis-Bouldin Score for clustering."""
        evaluator = DavisBouldinEvaluator()
        return evaluator.evaluate(df)


class SparkEngine:
    def __init__(self):
        """Initializes the Spark Engine."""
        self.spark_session = SparkSession.builder.getOrCreate()

    def read_csv(self, file_path):
        """Reads a CSV file and returns a DataFrame."""
        return self.spark_session.read.csv(file_path, header=True, inferSchema=True)

    def preprocess_data(self, df, feature_cols):
        """Selects the required feature columns."""
        return df.select(*feature_cols)

    def cluster_data(self, df, num_clusters):
        """Clusters the data using KMeans and returns the clustered DataFrame."""
        kmeans = KMeans(featuresCol="features", k=num_clusters, seed=1)
        model = kmeans.fit(df)
        return model.transform(df)

    def write_parquet(self, df, output_path):
        """Writes the DataFrame to a Parquet file."""
        df.write.parquet(output_path)

    def train_linear_regression(self, df, feature_cols, label_col):
        """Trains a Linear Regression model and returns predictions."""
        from pyspark.ml.regression import LinearRegression
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        lr = LinearRegression(featuresCol="features", labelCol=label_col)
        model = lr.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    def train_decision_tree_classifier(self, df, feature_cols, label_col):
        """Trains a Decision Tree Classifier and returns predictions."""
        from pyspark.ml.classification import DecisionTreeClassifier
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        dt = DecisionTreeClassifier(featuresCol="features", labelCol=label_col)
        model = dt.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    def train_random_forest_classifier(self, df, feature_cols, label_col):
        """Trains a Random Forest Classifier and returns predictions."""
        from pyspark.ml.classification import RandomForestClassifier
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        rf = RandomForestClassifier(featuresCol="features", labelCol=label_col)
        model = rf.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    def evaluate_model(self, predictions, label_col):
        """Evaluates a classification model using accuracy metric."""
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction",
                                                      metricName="accuracy")
        return evaluator.evaluate(predictions)

    def calculate_confusion_matrix(self, predictions, label_col):
        """Calculates and returns the confusion matrix for a classification model."""
        from pyspark.mllib.evaluation import MulticlassMetrics
        rdd = predictions.select(label_col, "prediction").rdd.map(tuple)
        metrics = MulticlassMetrics(rdd)
        return metrics.confusionMatrix().toArray()

    def train_model(self, df_preprocessed, label_col):
        pass

    def predict(self, model, df):
        pass
