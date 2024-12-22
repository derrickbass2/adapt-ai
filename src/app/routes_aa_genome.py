import numpy as np
from flask import Blueprint, request, jsonify

from adapt_backend.ml_models import AA_Genome  # Import the AA Genome class

# Initialize the Blueprint
aa_genome_bp = Blueprint("aa_genome", __name__)

# Shared `AA_Genome` instance (to maintain continuity across requests)
aa_genome_instance = AA_Genome()


@aa_genome_bp.route("/train", methods=["POST"])
def train_model():
    """
    Endpoint to train the AA_Genome model.
    Expects JSON with parameters for training.
    """
    try:
        # Extract and validate request JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Retrieve required parameters and their defaults
        dimensions = data.get("dimensions", 10)
        target_value = data.get("target_value", 0.5)
        pop_size = data.get("pop_size", 100)
        generations = data.get("generations", 1000)
        mutation_rate = data.get("mutation_rate", 0.01)

        # Validate parameters
        if dimensions <= 0 or pop_size <= 0 or generations <= 0:
            return jsonify({"error": "Dimensions, population size, and generations must be positive integers."}), 400
        if not (0 < mutation_rate <= 1):
            return jsonify({"error": "Mutation rate must be in the range (0, 1]."}), 400

        # Convert to numpy arrays if needed (example where features/labels might also be applicable)
        # Example: Ensure numeric/array types where applicable
        # np_data = np.array(data.get("some_field"))

        # Add logic for initializing or running the genetic algorithm
        # Here, `aa_genome_instance.train_AA_genome_model()` can be updated with real training logic
        aa_genome_instance.train_AA_genome_model(
            np.random.rand(pop_size, dimensions),  # Example random data
            np.ones(pop_size) * target_value  # Example labels
        )

        return jsonify({"message": "Model trained successfully"}), 200

    except ValueError as e:
        # Handle explicitly thrown validation errors
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@aa_genome_bp.route("/best_solution", methods=["GET"])
def get_best_solution():
    """
    Endpoint to get the best solution from the trained AA_Genome model.
    """
    try:
        # Retrieve the best solution from the shared AA_Genome instance
        best_solution = aa_genome_instance.get_best_solution()

        # Return solution as JSON response
        return jsonify({"best_solution": best_solution.tolist()}), 200

    except ValueError as e:
        # Handle errors related to no solution found
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@aa_genome_bp.route("/")
def home():
    """
    A simple home route to test the AA Genome API.
    """
    return jsonify({"message": "Welcome to the AA Genome API"}), 200
