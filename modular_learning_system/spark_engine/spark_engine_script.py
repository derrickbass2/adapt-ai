from pyspark.sql import SparkSession

DATAPATH = '/Users/derrickbass/Desktop/autonomod/datasets/hospitality/'
OUTPUTPATH = '/Users/derrickbass/Desktop/autonomod/datasets/cleaned_hospitality_data/'


def clean_data(spark, df):
    # Example data cleaning method
    filtered_df = df.filter(df['salary'] > 30000)
    return filtered_df


def main():
    spark = SparkSession.builder.appName('Spark Engine').getOrCreate()

    dfs = spark.read.csv(DATAPATH + '*.csv', header=True, inferSchema=True, multiline=True)

    cleaned_dfs = dfs.map(clean_data)

    combined_df = cleaned_dfs.reduce(lambda a, b: a.unionByName(b))
    combined_df.write.mode('overwrite').csv(OUTPUTPATH)


if __name__ == '__main__':


# mls/spark_engine/spark_engine_script.py
def run():
    # Your existing code here
    return "Spark Engine result"


if __name__ == "__main__":
    print(run())
    main()
