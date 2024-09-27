import sys

from flask import Flask, request, jsonify
from pyspark.sql import SparkSession

from spark_engine import SparkEngineUtils  # Ensure this module is correctly defined

sys.path.append('/Users/derrickbass/Public/adaptai/src/modular_learning_system/spark_engine')

app = Flask(__name__)

# Initialize SparkSession and SparkEngineUtils
spark = SparkSession.builder.appName('Spark Engine API').getOrCreate()
spark_engine = SparkEngineUtils(spark)


@app.route('/api/spark-engine/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    file_path = data['file_path']
    feature_cols = data['feature_cols']

    try:
        # Read the CSV file into a DataFrame
        df = spark_engine.read_csv(file_path)

        # Select only the specified feature columns
        processed_df = df.select(*feature_cols)

        # Convert DataFrame to Pandas for easier response handling
        pandas_df = spark_engine.convert_to_pandas(processed_df)

        # Return processed DataFrame in JSON format
        return jsonify(pandas_df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
