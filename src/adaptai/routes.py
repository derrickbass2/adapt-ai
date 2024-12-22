from http import HTTPStatus

from flask import Flask, request
from pydantic import BaseModel, EmailStr, ValidationError

from adaptai.response_utils import ResponseUtils
from adaptai.user_service import UserService

# Initialize Flask app and services
app = Flask(__name__)
user_service = UserService()


# Pydantic schemas for input validation
class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UpdateUserRequest(BaseModel):
    username: str = None
    email: EmailStr = None


# Routes

@app.route('/users', methods=['POST'])
def create_user():
    """Create a new user."""
    try:
        # Validate request data using CreateUserRequest schema
        req_body = CreateUserRequest(**request.get_json())
        response = user_service.create_user(req_body.dict())
        return ResponseUtils.send_response(HTTPStatus.CREATED, response)

    except ValidationError as e:
        # Return validation errors
        return ResponseUtils.send_response(HTTPStatus.BAD_REQUEST, {"error": e.errors()})

    except Exception as e:
        # Catch-all for unexpected errors
        return ResponseUtils.send_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


@app.route('/users/login', methods=['POST'])
def login_user():
    """Authenticate user."""
    try:
        # Validate login request data
        req_body = LoginRequest(**request.get_json())
        response = user_service.login_user(req_body.dict())
        return ResponseUtils.send_response(HTTPStatus.OK, response)

    except ValidationError as e:
        return ResponseUtils.send_response(HTTPStatus.BAD_REQUEST, {"error": e.errors()})

    except Exception as e:
        return ResponseUtils.send_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


@app.route('/users', methods=['GET'])
def get_users():
    """Retrieve a list of users."""
    try:
        response = user_service.get_users()
        return ResponseUtils.send_response(HTTPStatus.OK, response)

    except Exception as e:
        return ResponseUtils.send_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


@app.route('/users/<int:id>', methods=['GET'])
def get_user_by_id(id: int):
    """Retrieve a user by ID."""
    try:
        response = user_service.get_user_by_id(id)
        if response.get("message") == "User not found":
            return ResponseUtils.send_response(HTTPStatus.NOT_FOUND, response)

        return ResponseUtils.send_response(HTTPStatus.OK, response)

    except Exception as e:
        return ResponseUtils.send_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


@app.route('/users/<int:id>', methods=['PUT'])
def update_user(id: int):
    """Update an existing user."""
    try:
        # Validate update request data
        req_body = UpdateUserRequest(**request.get_json())
        response = user_service.update_user(id, req_body.dict())
        return ResponseUtils.send_response(HTTPStatus.OK, response)

    except ValidationError as e:
        return ResponseUtils.send_response(HTTPStatus.BAD_REQUEST, {"error": e.errors()})

    except Exception as e:
        return ResponseUtils.send_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


@app.route('/users/<int:id>', methods=['DELETE'])
def delete_user(id: int):
    """Delete a user."""
    try:
        response = user_service.delete_user(id)
        if response.get("message") == "User not found":
            return ResponseUtils.send_response(HTTPStatus.NOT_FOUND, response)

        return ResponseUtils.send_response(HTTPStatus.NO_CONTENT, None)

    except Exception as e:
        return ResponseUtils.send_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})


class UserController:
    pass
