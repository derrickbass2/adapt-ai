from flask import Blueprint, request, jsonify
from adapt_backend.ml_models import AA_Genome  # Import the model class

# Initialize the Blueprint
aa_genome_bp = Blueprint('aa_genome', __name__)

@aa_genome_bp.route('/train', methods=['POST'])
def train_model():
    """
    Endpoint to train the AA_Genome model.
    Expects JSON with parameters for training.
    """
    data = request.json
    dimensions = data.get('dimensions', 10)
    target_value = data.get('target_value', 0.5)
    pop_size = data.get('pop_size', 100)
    generations = data.get('generations', 1000)
    mutation_rate = data.get('mutation_rate', 0.01)

    # Initialize AA_Genome with provided parameters
    aa_genome = AA_Genome(
        dimensions=dimensions,
        target_value=target_value,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate
    )

    trained_model = aa_genome.train_AA_genome_model()

    return jsonify({"message": "Model trained successfully", "model": trained_model}), 200

@aa_genome_bp.route('/best_solution', methods=['GET'])
def get_best_solution():
    """
    Endpoint to get the best solution from the trained model.
    """
    aa_genome = AA_Genome()  # Ensure this instance is correctly managed
    best_solution = aa_genome.get_best_solution()

    return jsonify({"best_solution": best_solution}), 200

# Additional routes can be added here