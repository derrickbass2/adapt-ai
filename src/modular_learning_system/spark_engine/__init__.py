from typing import List, Any

import pandas as pd
from build.lib.modular_learning_system.spark_engine.spark_engine_script import clean_data
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

from spark_engine import SparkEngineUtils


def process_dataset(spark: SparkSession, file_path: str, output_path: str) -> None:
    """Read, clean, and write the processed data."""
    print(f"Processing {file_path}...")

    # Load data
    df = spark.read.option('header', 'true').csv(file_path)

    # Instantiate the utilities class for additional operations
    engine_utils = SparkEngineUtils(spark)

    # Clean and transform the data
    cleaned_df = clean_data(df)
    advanced_cleaned_df = engine_utils.advanced_cleaning(cleaned_df)

    # Optional: Run clustering
    feature_cols = ['col1', 'col2']  # Specify your feature columns
    clustered_df = engine_utils.run_kmeans(advanced_cleaned_df, feature_cols)

    # Save cleaned data
    engine_utils.write_output(clustered_df, output_path)
    print(f"Data saved to {output_path}")


__all__ = [
    'ColumnSelector',
    'MeanVectorStandardizer',
    'preprocess_data'
]


def _validate_input_dataframe(df: pd.DataFrame, column_names: List[str]) -> bool:
    """Validate input DataFrame columns."""
    if not all(col in df.columns for col in column_names):
        raise ValueError(f"DataFrame must contain columns: {column_names}")
    return True


def _extract_features(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """Extract features from the input DataFrame."""
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def _assemble_vector(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Create a vector column from specified features."""
    df['features'] = df[feature_columns].apply(lambda row: Vectors.dense(row), axis=1)
    return df


def _normalize_data(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Normalize the data using min-max scaling."""
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def _cluster_data(df: pd.DataFrame, feature_columns: List[str], k: int) -> pd.DataFrame:
    """Perform k-means clustering on the normalized data."""
    from pyspark.ml.clustering import KMeans

    spark = SparkSession.builder.getOrCreate()
    df_spark = spark.createDataFrame(df)

    vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    df_transformed = vec_assembler.transform(df_spark)

    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(df_transformed)
    predictions = model.transform(df_transformed)

    return predictions.toPandas()


def _train_model(df: pd.DataFrame, label_column: str, feature_columns: List[str]) -> Any:
    """Train a random forest classifier."""
    from pyspark.ml.classification import RandomForestClassifier

    spark = SparkSession.builder.getOrCreate()
    df_spark = spark.createDataFrame(df)

    vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    df_transformed = vec_assembler.transform(df_spark)

    rf = RandomForestClassifier(labelCol=label_column, seed=1)
    model = rf.fit(df_transformed)

    return model


def _predict(model: Any, df: pd.DataFrame, feature_columns: List[str]) -> pd.Series:
    """Predict labels using the trained model."""
    spark = SparkSession.builder.getOrCreate()
    df_spark = spark.createDataFrame(df)

    vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    df_transformed = vec_assembler.transform(df_spark)

    predictions = model.transform(df_transformed)
    return predictions.select('prediction').toPandas()['prediction']


def _evaluate_model(predictions: pd.Series, actual_labels: pd.Series) -> float:
    """Calculate the F1 score for the model."""
    from sklearn.metrics import f1_score
    return f1_score(actual_labels, predictions)


class ColumnSelector(Transformer):
    """
    Selects specified columns from the input DataFrame.
    """

    def __init__(self, selected_cols: List[str]):
        super().__init__()
        self._selected_cols = selected_cols

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.select(*self._selected_cols)


class MeanVectorStandardizer(Transformer):
    """
    Computes the mean and variance for each feature, and applies a transformation to center and scale the features.
    """

    def __init__(self, input_col: str, output_col: str):
        super().__init__()
        self._input_col = input_col
        self._output_col = output_col

    def _transform(self, df: DataFrame) -> DataFrame:
        vec_assembler = VectorAssembler(inputCols=[self._input_col], outputCol=self._output_col)
        transformed_df = vec_assembler.transform(df)

        stats = transformed_df.select(self._output_col).describe().toPandas()
        mu = stats[stats['summary'] == 'mean'][self._output_col].values[0]
        sigma = stats[stats['summary'] == 'stddev'][self._output_col].values[0]

        def standardize(v):
            return [(x - mu) / sigma for x in v]

        transf_vec = udf(standardize, ArrayType(DoubleType()))
        centered_df = transformed_df.withColumn(self._output_col, transf_vec(transformed_df[self._output_col]))

        return centered_df


def preprocess_data(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Load, preprocess, and return the cleaned DataFrame."""
    df = pd.read_csv(file_path, sep=sep)
    df.fillna(method='ffill', inplace=True)  # Fill missing values
    return df


class SparkEngine:
    def read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV file into a DataFrame."""
        return preprocess_data(file_path)

    def preprocess_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Preprocess the DataFrame for modeling."""
        df = _extract_features(df, categorical_cols=feature_cols)
        df = _normalize_data(df, feature_columns=feature_cols)
        return df

    def write_parquet(self, df: pd.DataFrame, output_path: str) -> None:
        """Write DataFrame to a Parquet file."""
        spark = SparkSession.builder.getOrCreate()
        df_spark = spark.createDataFrame(df)
        df_spark.write.parquet(output_path)

    def cluster_data(self, df: pd.DataFrame, feature_cols: List[str], num_clusters: int) -> pd.DataFrame:
        """Cluster data using k-means."""
        df = _normalize_data(df, feature_columns=feature_cols)
        return _cluster_data(df, feature_columns=feature_cols, k=num_clusters)

    def SparkEngine(self):
        pass
