import os
from datetime import timedelta
from typing import Optional

from flask import Flask, request, abort, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session as SQLAlchemySession
from werkzeug.security import safe_str_cmp

# Define the base class for SQLAlchemy models
Base = declarative_base()


# Define Role model
class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False, index=True)
    users = relationship("User", backref="role")


# Define User model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("roles.id"), default=1)
    created_at = Column(func.now(), server_default=func.now())
    modified_at = Column(func.now(), server_default=func.now(), onupdate=func.now())


# Define TokenBlocklist model
class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(func.now(), server_default=func.now())


# Define the UserService class for handling user-related operations
class UserService:
    def __init__(self, db_uri: str) -> None:
        self.engine = create_engine(db_uri)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @property
    def get_session(self) -> SQLAlchemySession:
        return self.Session()

    def get_revoked_token(self, jti: str) -> bool:
        with self.get_session as session:
            return session.query(TokenBlocklist).filter_by(jti=jti).first() is not None

    def authenticate_user(self, username: str) -> Optional[User]:
        with self.get_session as session:
            user = session.query(User).filter_by(username=username).first()
            if user and safe_str_cmp():
                return user
            return None

    def register_user(self, username: str, password: str, email: str, role_name: str):
        with self.get_session as session:
            role = session.query(Role).filter_by(name=role_name).first()
            if not role:
                role = Role(name=role_name)
                session.add(role)
                session.commit()
            user = User(username=username, password=password, email=email, role_id=role.id)
            session.add(user)
            session.commit()

    def revoke_token(self, jti: str):
        with self.get_session as session:
            blocklist = TokenBlocklist(jti=jti)
            session.add(blocklist)
            session.commit()

    def check_if_token_in_blocklist(self, jti: str) -> bool:
        with self.get_session as session:
            return session.query(TokenBlocklist).filter_by(jti=jti).first() is not None


def identity_lookup(payload):
    user_id = payload["sub"]
    user_svc = UserService(os.getenv('DATABASE_URI'))
    with user_svc.get_session as session:
        return session.query(User).filter_by(id=user_id).first()


# Initialize Flask app and configure JWT
app = Flask(__name__)
app.config['SECRET_KEY'] = b'Pc\xe2\xe5a@\x96\xa6\xd7\xaa2\xfb\xde\xd1U!\x07\x10\x00\xfdDlS\x8b'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
jwt = JWTManager(app)


# Protect routes that require JWT token
@app.before_request
def protect_bearer_only():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
        abort(401)


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    role_name = data.get('role_name', 'User')  # Default role name
    user_svc = UserService(os.getenv('DATABASE_URI'))
    user_svc.register_user(username, password, email, role_name)
    return jsonify(message="User registered successfully"), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    data.get('password')
    user_svc = UserService(os.getenv('DATABASE_URI'))
    user = user_svc.authenticate_user(username)
    if user:
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token), 200
    return jsonify(message="Invalid credentials"), 401


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt_identity()
    if not jti:
        return jsonify(message="JWT not found"), 400
    user_svc = UserService(os.getenv('DATABASE_URI'))
    user_svc.revoke_token(jti)
    return jsonify(message="Successfully logged out"), 200


@jwt.token_in_blocklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService(os.getenv('DATABASE_URI'))
    return user_svc.check_if_token_in_blocklist(jti)


if __name__ == "__main__":
    app.run()
