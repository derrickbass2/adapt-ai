from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# Paths to the datasets
RETAIL_DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/retaildatasets/'
HOSPITALITY_DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/hospitalitydatasets/'
PSYCH_DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/psychdatasets/'
DOW_JONES_DATAPATH = '/Users/derrickbass/Public/adaptai/datasets/dow+jones+index/'
RETAIL_OUTPUTPATH = '/Users/derrickbass/Public/adaptai/datasets/retaildatasets/cleaned_retail_datasets/'
HOSPITALITY_OUTPUTPATH = '/Users/derrickbass/Public/adaptai/datasets/hospitalitydatasets/cleaned_hospitality_datasets/'
PSYCH_OUTPUTPATH = '/Users/derrickbass/Public/adaptai/datasets/psychdatasets/cleaned_psych_datasets/'
DOW_JONES_OUTPUTPATH = '/Users/derrickbass/Public/adaptai/datasets/dow+jones+index/cleaned_dow_jones_datasets/'

# Custom cleaning functions
def clean_retail_data(df, file_name):
    print(f"Schema of DataFrame '{file_name}':")
    df.printSchema()

    print(f"First few rows of DataFrame '{file_name}':")
    df.show(5)
    
    if file_name == 'blinkit_retail.csv':
        df = df.dropDuplicates().dropna()
    elif file_name == 'olist_customers_dataset.csv':
        df = df.dropDuplicates().dropna(subset=['customer_id'])
    elif file_name == 'olist_order_items_dataset.csv':
        df = df.filter((col('price') > 0)).dropna()
    elif file_name == 'olist_orders_dataset.csv':
        df = df.withColumn('order_purchase_timestamp', to_date(col('order_purchase_timestamp'))).dropna()
    elif file_name == 'olist_sellers_dataset.csv':
        df = df.dropDuplicates().dropna(subset=['seller_id'])
    elif file_name == 'product_category_name_translation.csv':
        df = df.dropDuplicates().dropna()
    elif file_name == 'shopping_trends.csv':
        df = df.dropDuplicates().dropna()
    return df

def clean_hospitality_data(df, file_name):
    print(f"Schema of DataFrame '{file_name}':")
    df.printSchema()

    print(f"First few rows of DataFrame '{file_name}':")
    df.show(5)

    if file_name == 'HospitalityEmployees.csv':
        df = df.filter(col('salary') > 0).dropna(subset=['employee_id'])
    elif file_name == 'Cocktaildatasets/all_drinks.csv':
        df = df.dropDuplicates().dropna()
    elif file_name == 'Cocktaildatasets/data_cocktails.csv':
        df = df.dropDuplicates().dropna()
    elif file_name == 'Cocktaildatasets/hotaling_cocktails - Cocktails.csv':
        df = df.dropDuplicates().dropna()
    return df

def clean_psych_data(df, file_name):
    print(f"Schema of DataFrame '{file_name}':")
    df.printSchema()

    print(f"First few rows of DataFrame '{file_name}':")
    df.show(5)

    if file_name == 'garments_worker_productivity.csv':
        df = df.filter(col('actual_productivity') > 0).dropna()
    elif file_name == 'shopping_behavior_updated.csv':
        df = df.dropDuplicates().dropna()
    return df

def clean_dow_jones_data(df, file_name):
    print(f"Schema of DataFrame '{file_name}':")
    df.printSchema()

    print(f"First few rows of DataFrame '{file_name}':")
    df.show(5)

    if file_name == 'edstats-csv-zip-32-mb-/EdStatsCountry-Series.csv':
        df = df.dropna(subset=['Country Code']).dropDuplicates()
    elif file_name == 'edstats-csv-zip-32-mb-/EdStatsSeries.csv':
        df = df.dropna(subset=['Series Code']).dropDuplicates()
    elif file_name == 'statlog+german+credit+data/EdStatsCountry.csv':
        df = df.dropna(subset=['Country Name']).dropDuplicates()
    return df

def main():
    spark = SparkSession.builder.appName('Spark Engine').getOrCreate()

    # Process retail datasets
    retail_files = [
        'blinkit_retail.csv',
        'olist_customers_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_orders_dataset.csv',
        'olist_sellers_dataset.csv',
        'product_category_name_translation.csv',
        'shopping_trends.csv'
    ]
    for file in retail_files:
        df = spark.read.csv(RETAIL_DATAPATH + file, header=True, inferSchema=True)
        cleaned_df = clean_retail_data(df, file)
        cleaned_df.write.mode('overwrite').csv(RETAIL_OUTPUTPATH + file.replace('.csv', '_cleaned'))

    # Process hospitality datasets
    hospitality_files = [
        'HospitalityEmployees.csv',
        'Cocktaildatasets/all_drinks.csv',
        'Cocktaildatasets/data_cocktails.csv',
        'Cocktaildatasets/hotaling_cocktails - Cocktails.csv'
    ]
    for file in hospitality_files:
        df = spark.read.csv(HOSPITALITY_DATAPATH + file, header=True, inferSchema=True)
        cleaned_df = clean_hospitality_data(df, file)
        cleaned_df.write.mode('overwrite').csv(HOSPITALITY_OUTPUTPATH + file.replace('.csv', '_cleaned'))

    # Process psych datasets
    psych_files = [
        'garments_worker_productivity.csv',
        'shopping_behavior_updated.csv'
    ]
    for file in psych_files:
        df = spark.read.csv(PSYCH_DATAPATH + file, header=True, inferSchema=True)
        cleaned_df = clean_psych_data(df, file)
        cleaned_df.write.mode('overwrite').csv(PSYCH_OUTPUTPATH + file.replace('.csv', '_cleaned'))

    # Process Dow Jones datasets
    dow_jones_files = [
        'edstats-csv-zip-32-mb-/EdStatsCountry-Series.csv',
        'edstats-csv-zip-32-mb-/EdStatsSeries.csv',
        'statlog+german+credit+data/EdStatsCountry.csv'
    ]
    for file in dow_jones_files:
        df = spark.read.csv(DOW_JONES_DATAPATH + file, header=True, inferSchema=True)
        cleaned_df = clean_dow_jones_data(df, file)
        cleaned_df.write.mode('overwrite').csv(DOW_JONES_OUTPUTPATH + file.replace('.csv', '_cleaned'))

if __name__ == '__main__':
    main()