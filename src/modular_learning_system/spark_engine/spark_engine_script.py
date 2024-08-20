from pyspark.sql import SparkSession
from pyspark.sql.functions import col

DATAPATH = '/Users/derrickbass/Desktop/autonomod/datasets/hospitality/'
OUTPUTPATH = '/Users/derrickbass/Desktop/autonomod/datasets/cleaned_hospitality_data/'


def clean_data(df):
    """
    Example data cleaning method that filters out rows with a salary less than or equal to 30,000.
    """
    filtered_df = df.filter(col('salary') > 30000)
    return filtered_df


def main():
    spark = SparkSession.builder.appName('Spark Engine').getOrCreate()

    # Read multiple CSV files from a directory
    dfs = spark.read.option('header', 'true').csv(DATAPATH + '*.csv', inferSchema=True, multiLine=True)

    # Apply data cleaning
    cleaned_dfs = dfs.map(clean_data)

    # Combine cleaned dataframes
    combined_df = cleaned_dfs.reduce(lambda a, b: a.unionByName(b))

    # Write combined DataFrame to Parquet format
    combined_df.write.mode('overwrite').parquet(OUTPUTPATH)

    # Stop the Spark session
    spark.stop()


if __name__ == '__main__':
    main()