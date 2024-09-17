import logging
import os

from pyspark.sql import DataFrame, SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths to the datasets and output directory
DATAPATHS = [
    '/Users/derrickbass/Public/adaptai/datasets/retail/',
    '/Users/derrickbass/Public/adaptai/datasets/finance/ed_stats_country.csv',
    '/Users/derrickbass/Public/adaptai/datasets/finance/ed_stats_country_series.csv',
    '/Users/derrickbass/Public/adaptai/datasets/finance/ed_stats_series.csv',
    '/Users/derrickbass/Public/adaptai/datasets/hospitality/all_drinks.csv',
    '/Users/derrickbass/Public/adaptai/datasets/hospitality/data_cocktails.csv',
    '/Users/derrickbass/Public/adaptai/datasets/hospitality/hospitality_employees.csv',
    '/Users/derrickbass/Public/adaptai/datasets/hospitality/hotaling_cocktails.csv',
    '/Users/derrickbass/Public/adaptai/datasets/psych/garments_worker_productivity.csv',
    '/Users/derrickbass/Public/adaptai/datasets/psych/shopping_behavior_updated.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/blinkit_retail.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/olist_customers_dataset.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/olist_order_items_dataset.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/olist_orders_dataset.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/olist_sellers_dataset.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/product_category_name_translation.csv',
    '/Users/derrickbass/Public/adaptai/datasets/retail/shopping_trends.csv'
]

OUTPUTPATH = '/Users/derrickbass/Public/adaptai/datasets/cleaned_data/'


def clean_data(df: DataFrame) -> DataFrame:
    """
    Example data cleaning method. This function can be expanded to fit various data cleaning needs.
    """
    logger.info("Cleaning data...")
    if 'salary' in df.columns:
        filtered_df = df.filter(df['salary'] > 30000)
    else:
        filtered_df = df
    return filtered_df


def process_dataset(spark: SparkSession, file_path: str, output_path: str) -> None:
    """Read, clean, and write the processed data."""
    logger.info(f"Processing {file_path}...")

    # Load data
    df = spark.read.option('header', 'true').csv(file_path)

    # Clean the data
    cleaned_df = clean_data(df)

    # Define output file path
    output_file = os.path.join(output_path, os.path.basename(file_path).replace('.csv', '.parquet'))

    # Save cleaned data
    cleaned_df.write.parquet(output_file)
    logger.info(f"Data saved to {output_file}")


def main():
    # Initialize Spark Session
    spark = SparkSession.builder.appName('Spark Engine').getOrCreate()

    # Ensure output directory exists
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)

    # Process each dataset
    for data_path in DATAPATHS:
        if os.path.isfile(data_path):  # If the path is a single file
            process_dataset(spark, data_path, OUTPUTPATH)
        else:  # If the path is a directory, process all CSVs in it
            for file in os.listdir(data_path):
                if file.endswith('.csv'):
                    process_dataset(spark, os.path.join(data_path, file), OUTPUTPATH)

    # Stop Spark Session
    spark.stop()


if __name__ == "__main__":
    main()
