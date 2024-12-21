import os

import requests
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)
port = int(os.getenv('PORT', 3000))  # Use port from environment or default to 3000


@app.route('/api/data', methods=['POST'])
def process_data():
    """
    API route to process data by chaining multiple services:
    - Preprocess data
    - Perform clustering
    - Process with AA Genome
    - Analyze with Neurotech network

    Expects:
    - file_path (str): The path to the input data file
    - feature_cols (list): List of feature columns to process
    - num_clusters (int): Number of clusters for the clustering algorithm
    """
    # Validate and parse JSON data
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON payload'}), 400

        file_path = data.get('file_path')
        feature_cols = data.get('feature_cols')
        num_clusters = data.get('num_clusters')

        if not file_path or not feature_cols or not num_clusters:
            return jsonify({
                'error': 'Invalid input: file_path, feature_cols, and num_clusters are required.'
            }), 400
    except Exception as e:
        return jsonify({'error': f'Invalid request: {str(e)}'}), 400

    # Define service URLs
    preprocess_url = 'http://localhost:3001/api/spark-engine/preprocess'
    cluster_url = 'http://localhost:3001/api/spark-engine/cluster'
    aa_genome_url = 'http://localhost:3002/api/aa-genome/process'
    neurotech_network_url = 'http://localhost:3003/api/neurotech-network/analyze'

    try:
        # Step 1: Preprocess the data
        post_request(
            preprocess_url, {'file_path': file_path, 'feature_cols': feature_cols}
        )

        # Step 2: Perform clustering
        cluster_response = post_request(
            cluster_url, {'file_path': file_path, 'num_clusters': num_clusters}
        )

        # Step 3: Process data with AA Genome
        aa_genome_response = post_request(
            aa_genome_url, cluster_response.json()
        )

        # Step 4: Analyze the data with Neurotech Network
        neurotech_network_response = post_request(
            neurotech_network_url, aa_genome_response.json()
        )

        # Return the final result from Neurotech Network
        return jsonify(neurotech_network_response.json()), 200

    except requests.exceptions.HTTPError as http_err:
        return jsonify({'error': f'HTTP error occurred: {http_err}'}), 500
    except requests.exceptions.RequestException as req_err:
        return jsonify({'error': f'Connection error occurred: {req_err}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def post_request(url, payload):
    """
    Helper function to perform POST requests with error handling.

    Args:
        url (str): The service endpoint URL.
        payload (dict): The JSON payload to send.

    Returns:
        Response object: The response from the target service.
    """
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise HTTPError for bad statuses (4xx and 5xx)
    return response


if __name__ == '__main__':
    # Run the Flask app on the specified port
    app.run(port=port)
