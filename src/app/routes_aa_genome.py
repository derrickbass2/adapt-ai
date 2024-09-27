from flask import Blueprint, request, jsonify

from adapt_backend.ml_models import AA_Genome  # Import the AA Genome class

# Initialize the Blueprint
aa_genome_bp = Blueprint('aa_genome', __name__)


@aa_genome_bp.route('/train', methods=['POST'])
def train_model():
    """
    Endpoint to train the AA_Genome model.
    Expects JSON with parameters for training.
    """
    try:
        data = request.json
        data.get('dimensions', 10)
        data.get('target_value', 0.5)
        data.get('pop_size', 100)
        data.get('generations', 1000)
        data.get('mutation_rate', 0.01)

        # Initialize AA_Genome and train it with provided parameters
        aa_genome = AA_Genome()
        trained_model = aa_genome.train_AA_genome_model()

        return jsonify({"message": "Model trained successfully", "model": trained_model}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@aa_genome_bp.route('/best_solution', methods=['GET'])
def get_best_solution():
    """
    Endpoint to get the best solution from the trained AA_Genome model.
    """
    try:
        aa_genome = AA_Genome()  # Ensure this instance is correctly managed
        best_solution = aa_genome.get_best_solution()

        return jsonify({"best_solution": best_solution}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@aa_genome_bp.route('/')
def home():
    """
    A simple home route to test the AA Genome API.
    """
    return jsonify({"message": "Welcome to the AA Genome API"}), 200
