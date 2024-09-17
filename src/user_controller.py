from http import HTTPStatus

from flask import Flask, request

from src.response_utils import ResponseUtils
from .user_service import UserService

app = Flask(__name__)
user_service = UserService()


@app.route('/users', methods=['POST'])
def create_user():
    req_body = request.get_json()
    response = user_service.create_user(req_body)
    return ResponseUtils.send_response(HTTPStatus.CREATED, response)


@app.route('/users/login', methods=['POST'])
def login_user():
    req_body = request.get_json()
    response = user_service.login_user(req_body)
    return ResponseUtils.send_response(HTTPStatus.OK, response)


@app.route('/users', methods=['GET'])
def get_users():
    response = user_service.get_users()
    return ResponseUtils.send_response(HTTPStatus.OK, response)


@app.route('/users/<int:id>', methods=['GET'])
def get_user_by_id(id: int):
    response = user_service.get_user_by_id(id)
    return ResponseUtils.send_response(HTTPStatus.OK, response)


@app.route('/users/<int:id>', methods=['PUT'])
def update_user(id: int):
    req_body = request.get_json()
    response = user_service.update_user(id, req_body)
    return ResponseUtils.send_response(HTTPStatus.OK, response)


@app.route('/users/<int:id>', methods=['DELETE'])
def delete_user(id: int):
    user_service.delete_user(id)
    return ResponseUtils.send_response(HTTPStatus.NO_CONTENT, None)


class UserController:
    def __init__(self, user_repository):
        self.user_repository = user_repository

    def get_users(self):
        return self.user_repository.get_users()

    def get_user_by_id(self, id):
        return self.user_repository.get_user_by_id(id)

    def create_user(self, user_data):
        user_schema = UserSchema(**user_data)
        return self.user_repository.create_user(user_schema)

    def update_user(self, id, user_data):
        user_schema = UserSchema(**user_data)
        return self.user_repository.update_user(id, user_schema)

    def delete_user(self, id):
        self.user_repository.delete_user(id)
