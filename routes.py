from flask import Blueprint, jsonify

bp = Blueprint('routes', __name__)


@bp.route('/')
def index():
    return "Hello, world!"


@bp.route('/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from the /hello route!")


def routes_aa_genome():
    """
    """
    return None


def routes_neurotech_network():
    return None


def routes_spark_engine():
    return None


def routes_aa_genome():
    return None
