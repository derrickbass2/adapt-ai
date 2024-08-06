import sys
import os

sys.path.insert(0, os.path.expanduser("~"))
sys.path.append(os.path.join(os.path.expanduser("~"), "nomad"))

from nomad import fetch_data

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def fetch_data_endpoint():
    raw_data = fetch_data()
    processed_data = ... # Process the data using your Nomad functions
    return jsonify({'result': processed_data})

if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)