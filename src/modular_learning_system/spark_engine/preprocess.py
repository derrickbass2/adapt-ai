import sys

from flask import Flask, request, jsonify
from spark_engine import SparkEngine  # Ensure this module is correctly defined

# Add the path where `spark_engine` module is located
sys.path.append('/Users/derrickbass/Public/adaptai/src/modular_learning_system/spark_engine')

app = Flask(__name__)
spark_engine = SparkEngine()


@app.route('/api/spark-engine/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    file_path = data['file_path']
    feature_cols = data['feature_cols']

    # Read the CSV file into a DataFrame
    df = spark_engine.read_csv(file_path)

    # Preprocess the DataFrame
    preprocessed_df = spark_engine.preprocess_data(df, feature_cols)

    # Convert DataFrame to a JSON-compatible dictionary
    result = preprocessed_df.toPandas().to_dict(orient='records')

    # Return processed data as JSON
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=3001)  # Ensure the port matches your server setup
