from pyspark.sql import SparkSession

DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/hospitality/'
OUTPUTPATH = '/Users/derrickbass/Public/adaptai/datasets/cleaned_hospitality_data/'


def clean_data(df):
    # Example data cleaning method
    filtered_df = df.filter(df['salary'] > 30000)
    return filtered_df


def main():
    spark = SparkSession.builder.appName('Spark Engine').getOrCreate()

    dfs = spark.read.csv(DATAPATH + '*.csv', header=True, inferSchema=True)

    cleaned_dfs = dfs.map(clean_data)

    combined_df = cleaned_dfs.reduce(lambda a, b: a.unionByName(b))
    combined_df.write.mode('overwrite').csv(OUTPUTPATH)


if __name__ == '__main__':
    main()