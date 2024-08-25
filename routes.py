from flask import Blueprint, jsonify

bp = Blueprint('routes', __name__)


@bp.route('/')
def index():
    return "Hello, world!"


@bp.route('/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from the /hello route!")
