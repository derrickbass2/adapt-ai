from adaptai import fetch_data  # Ensure this import is correct
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/api/data', methods=['GET'])
def fetch_data_endpoint():
    try:
        raw_data = fetch_data()
        processed_data = process_data(raw_data)  # Use a placeholder function to process data
        return jsonify({'result': processed_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_data(raw_data):
    # Placeholder function to process raw data
    # Replace with actual processing logic
    return raw_data


if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Run the Flask app on port 5000
