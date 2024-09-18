import sys

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession

from spark_engine import SparkEngineUtils  # Ensure this module is correctly defined

spark = SparkSession.builder.appName('Spark Engine API').getOrCreate()
spark_engine = SparkEngineUtils(spark)

sys.path.append('/Users/derrickbass/Public/adaptai/src/modular_learning_system/spark_engine')

app = Flask(__name__)
spark_engine = SparkEngineUtils(spark)  # Initialize with a valid SparkSession


@app.route('/api/spark-engine/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    file_path = data['file_path']
    var = data['feature_cols']

    # Initialize SparkSession and SparkEngineUtils
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('Spark Engine API').getOrCreate()
    spark_engine = SparkEngineUtils(spark)

    try:
        # Read the CSV file into a DataFrame
        df = spark_engine.read_csv(file_path)

        # Preprocess the DataFrame
        preprocessed_df = spark_engine.advanced_cleaning(df)  # Use the appropriate method for preprocessing

        # Convert DataFrame to a JSON-compatible dictionary
        result = preprocessed_df.toPandas().to_dict(orient='records')

        # Return processed data as JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        spark.stop()


if __name__ == '__main__':
    app.run(debug=True, port=3001)  # Ensure the port matches your server setup
