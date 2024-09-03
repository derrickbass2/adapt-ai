from typing import List, Any

import pandas as pd
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.pandas._typing import DataFrameLike
from pyspark.sql.types import ArrayType, DoubleType

__all__ = [
    'ColumnSelector',
    'MeanVectorStandardizer',
    'preprocess_data'
]


def _validate_input_dataframe(df: pd.DataFrame, column_names: List[str]) -> bool:
    """Validate input DataFrame columns."""
    if not all(col in df.columns for col in column_names):
        raise ValueError("DataFrame must contain columns: {column_names}")
    return True


def _extract_features(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """Extract features from the input DataFrame."""
    # Example implementation: One-hot encode categorical columns and normalize numerical columns
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


def _cluster_data(df: pd.DataFrame, feature_columns: List[str], k: int) -> DataFrameLike:
    """Perform k-means clustering on the normalized data."""
    from pyspark.ml.clustering import KMeans
    from pyspark.sql import SparkSession

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
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    df_spark = spark.createDataFrame(df)

    vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    df_transformed = vec_assembler.transform(df_spark)

    rf = RandomForestClassifier(labelCol=label_column, seed=1)
    model = rf.fit(df_transformed)

    return model


def _predict(model: Any, df: pd.DataFrame, feature_columns: List[str]) -> pd.Series:
    """Predict labels using the trained model."""
    from pyspark.sql import SparkSession

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

    @property
    def _transformFunc(self):
        def transform(df: DataFrame) -> DataFrame:
            return df.select(*self._selected_cols)

        return transform


class MeanVectorStandardizer(Transformer):
    """
    Computes the mean and variance for each feature, and applies a transformation to center and scale the features.
    """

    def __init__(self, input_col: str, output_col: str):
        super().__init__()
        self._input_col = input_col
        self._output_col = output_col

    @property
    def _transformFunc(self):
        def transform(df: DataFrame) -> DataFrame:
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

        return transform


def preprocess_data(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Load, preprocess, and return the cleaned DataFrame."""
    df = pd.read_csv(file_path, sep=sep)
    # Example preprocessing steps: fill missing values
    df.fillna(method='ffill', inplace=True)
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
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        df_spark = spark.createDataFrame(df)
        df_spark.write.parquet(output_path)

    def cluster_data(self, df: pd.DataFrame, feature_cols: List[str], num_clusters: int) -> DataFrameLike:
        """Cluster data using k-means."""
        df = _normalize_data(df, feature_columns=feature_cols)
        return _cluster_data(df, feature_columns=feature_cols, k=num_clusters)
