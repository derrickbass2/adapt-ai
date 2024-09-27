import os

import requests
from flask import Flask, request

app = Flask(__name__)
port = int(os.getenv('PORT', 3000))


@app.route('/api/data', methods=['POST'])
def process_data():
    data = request.get_json()
    file_path = data.get('file_path')
    feature_cols = data.get('feature_cols')
    num_clusters = data.get('num_clusters')

    preprocess_url = 'http://localhost:3001/api/spark-engine/preprocess'
    cluster_url = 'http://localhost:3001/api/spark-engine/cluster'
    aa_genome_url = 'http://localhost:3002/api/aa-genome/process'
    neurotech_network_url = 'http://localhost:3003/api/neurotech-network/analyze'

    try:
        requests.post(preprocess_url, json={'file_path': file_path, 'feature_cols': feature_cols})
        cluster_response = requests.post(cluster_url, json={'file_path': file_path, 'num_clusters': num_clusters})
        aa_genome_response = requests.post(aa_genome_url, json=cluster_response.json())
        neurotech_network_response = requests.post(neurotech_network_url, json=aa_genome_response.json())

        return neurotech_network_response.json(), 200
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(port=port)


def spark_engine():
    return None
