from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession


class SparkEngineUtils:
    """
    Utility class for common Spark data conversion and feature processing tasks.
    """

    @staticmethod
    def convert_to_pandas(df):
        """
        Convert a Spark DataFrame to a Pandas DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame to convert.

        Returns
        -------
        pandas.DataFrame
            The converted Pandas DataFrame.
        """
        return df.toPandas()

    @staticmethod
    def convert_to_spark(df, spark):
        """
        Convert a Pandas DataFrame to a Spark DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The Pandas DataFrame to convert.
        spark : pyspark.sql.SparkSession
            The active Spark session.

        Returns
        -------
        pyspark.sql.DataFrame
            The converted Spark DataFrame.
        """
        return spark.createDataFrame(df)

    @staticmethod
    def vectorize_features(df, feature_cols):
        """
        Assemble specified columns into a feature vector for machine learning models.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame containing the feature columns.
        feature_cols : list of str
            List of column names to be used as features.

        Returns
        -------
        pyspark.sql.DataFrame             with an additional 'features' column containing vectorized features.
        """
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        return assembler.transform(df)

    @staticmethod
    def calculate_silhouette_score(df, prediction_col):
        """
        Calculate the Silhouette score for clustering results.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame with clustering predictions.
        prediction_col : str
            The name of the column containing clustering predictions.

        Returns
        -------
        float
            The Silhouette score for the clustering results.
        """
        evaluator = ClusteringEvaluator(predictionCol=prediction_col)
        return evaluator.evaluate(df)


class SparkEngine:
    """
    Main class for Spark engine operations, including data ingestion, preprocessing,
    and machine learning model training.
    """

    def __init__(self, spark=None):
        """
        Initialize the SparkEngine with an optional existing Spark session.

        Parameters
        ----------
        spark : pyspark.sql.SparkSession, optional
            The existing Spark session to use. If not provided, a new session is created.
        """
        self.spark_session = spark if spark else SparkSession.builder.getOrCreate()

    def get_spark_session(self):
        """
        Get the current Spark session.

        Returns
        -------
        pyspark.sql.SparkSession
            The active Spark session.
        """
        return self.spark_session

    def close_spark_session(self):
        """
        Close the current Spark session.
        """
        if self.spark_session:
            self.spark_session.stop()

    def read_csv(self, file_path):
        """
        Read a CSV file into a Spark DataFrame.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.

        Returns
        -------
        pyspark.sql.DataFrame
            The loaded Spark DataFrame.
        """
        return self.spark_session.read.csv(file_path, header=True, inferSchema=True)

    @staticmethod
    def preprocess_data(df, feature_cols):
        """
        Select specified feature columns from the DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame to preprocess.
        feature_cols : list of str
            List of columns to be selected as features.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing only the specified feature columns.
        """
        return df.select(*feature_cols)

    @staticmethod
    def cluster_data(df, num_clusters):
        """
        Perform K-means clustering on the DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame with features for clustering.
        num_clusters : int
            The number of clusters to use in K-means.

        Returns
        -------
        pyspark.sql.DataFrame             with an additional column containing cluster predictions.
        """
        kmeans = KMeans(k=num_clusters, seed=1)
        model = kmeans.fit(df)
        return model.transform(df)

    @staticmethod
    def write_parquet(df, output_path):
        """
        Write a Spark DataFrame to Parquet format.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The DataFrame to write.
        output_path : str
            The file path where the Parquet file will be saved.

        Returns
        -------
        None
        """
        df.write.parquet(output_path)

    @staticmethod
    def train_linear_regression(df, feature_cols, label_col):
        """
        Train a Linear Regression model on the DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame containing features and labels.
        feature_cols : list of str
            List of columns to be used as features.
        label_col : str
            The column name containing the label.

        Returns
        -------
        pyspark.sql.DataFrame             with predictions made by the model.
        """
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        lr = LinearRegression(labelCol=label_col)
        model = lr.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    @staticmethod
    def train_decision_tree_classifier(df, feature_cols, label_col):
        """
        Train a Decision Tree classifier on the DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame containing features and labels.
        feature_cols : list of str
            List of columns to be used as features.
        label_col : str
            The column name containing the label.

        Returns
        -------
        pyspark.sql.DataFrame             with predictions made by the model.
        """
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        dt = DecisionTreeClassifier(labelCol=label_col)
        model = dt.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    @staticmethod
    def train_random_forest_classifier(df, feature_cols, label_col):
        """
        Train a Random Forest classifier on the DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The Spark DataFrame containing features and labels.
        feature_cols : list of str
            List of columns to be used as features.
        label_col : str
            The column name containing the label.

        Returns
        -------
        pyspark.sql.DataFrame             with predictions made by the model.
        """
        df_vectorized = SparkEngineUtils.vectorize_features(df, feature_cols)
        rf = RandomForestClassifier(labelCol=label_col)
        model = rf.fit(df_vectorized)
        return model.transform(df_vectorized).select("features", "prediction", label_col)

    @staticmethod
    def evaluate_model(predictions, label_col):
        """
        Evaluate the accuracy of a classification model using a MulticlassClassificationEvaluator.

        Parameters
        ----------
        predictions : pyspark.sql.DataFrame             containing the predicted and actual labels.
        label_col : str
            The column containing the true label values.

        Returns
        -------
        float
            The accuracy of the model.
        """
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
        return evaluator.evaluate(predictions)

    @staticmethod
    def calculate_confusion_matrix(predictions, label_col):
        """
        Calculate the confusion matrix for a classification model.

        Parameters
        ----------
        predictions : pyspark.sql.DataFrame             containing the predicted and actual labels.
        label_col : str
            The column containing the true label values.

        Returns
        -------
        numpy.ndarray
            The confusion matrix as a 2D array.
        """
        rdd = predictions.select(label_col, "prediction").rdd.map(lambda row: (row[label_col], row["prediction"]))
        metrics = MulticlassMetrics(rdd)
        return metrics.confusionMatrix().toArray()
