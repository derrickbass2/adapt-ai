import sys
sys.path.append('/Users/derrickbass/Desktop/Autonomod/src')  # Add the path to the missing package

from flask import Blueprint, request
from http import HTTPStatus

from .user_services import UserService  # Fix the import statement

from src import ResponseUtils

api = Blueprint('api', __name__)

user_service = UserService()

@api.route('/users', methods=['POST'])
def create_user():
    req_body = request.get_json()
    response = user_service.create_user(req_body)
    return ResponseUtils.send_response(HTTPStatus.CREATED, response)

@api.route('/users/login', methods=['POST'])
def login_user():
    req_body = request.get_json()
    response = user_service.login_user(req_body)
    return ResponseUtils.send_response(HTTPStatus.OK, response)

@api.route('/users', methods=['GET'])
def get_users():
    response = user_service.get_users()
    return ResponseUtils.send_response(HTTPStatus.OK, response)

@api.route('/users/<int:id>', methods=['GET'])
def get_user_by_id(id: int):
    response = user_service.get_user_by_id(id)
    return ResponseUtils.send_response(HTTPStatus.OK, response)

@api.route('/users/<int:id>', methods=['PUT'])
def update_user(id: int):
    req_body = request.get_json()
    response = user_service.update_user(id, req_body)
    return ResponseUtils.send_response(HTTPStatus.OK, response)

@api.route('/users/<int:id>', methods=['DELETE'])
def delete_user(id: int):
    response = user_service.delete_user(id)
<<<<<<< HEAD:src/user_controller 2.py
    return ResponseUtils.send_response(HTTPStatus.NO_CONTENT, None)
=======
# src/user_controller.py
from user_repository import UserRepository

class UserController:
    def __init__(self):
        self.repo = UserRepository()

    def create_user(self, user_id, user_data):
        self.repo.add_user(user_id, user_data)
        return {"status": "success", "message": "User created"}

    def fetch_user(self, user_id):
        user = self.repo.get_user(user_id)
        if user:
            return {"status": "success", "data": user}
        else:
            return {"status": "error", "message": "User not found"}
eturn ResponseUtils.send_response(HTTPStatus.NO_CONTENT, None)
>>>>>>> c912b46a (cleaning up commits so that I can pull and clean out branches):src/user_controller.py
