from datetime import timedelta

from flask import Flask, jsonify, request, abort
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.exc import SQLAlchemyError  # Import specific exception
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from werkzeug.security import safe_str_cmp

Base = declarative_base()


class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False, index=True)
    users = relationship("User", backref="role")


class User(Base):
    query = None
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("roles.id"), default=1)
    created_at = Column(func.now(), default=None)
    modified_at = Column(func.now(), onupdate=None)


class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(func.now(), default=None)


class UserService:
    def __init__(self, db_uri):
        engine = create_engine(db_uri)
        Base.metadata.create_all(bind=engine)
        self._session = sessionmaker(bind=engine)()

    @staticmethod
    def identity_lookup(payload):
        user_id = payload["sub"]
        return User.query.filter_by(id=user_id).first()

    def register_user(self, username, password, email, role_id):
        user = User(username=username, password=password, email=email, role_id=role_id)
        try:
            self._session.add(user)
            self._session.commit()
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error occurred while inserting user: {e}")
            self._session.rollback()  # Rollback in case of error

    def authenticate_user(self, username):
        user = self._session.query(User).filter_by(username=username).first()
        if user and safe_str_cmp():
            return user
        return None

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error occurred while inserting blacklisted token: {e}")
            self._session.rollback()  # Rollback in case of error

    def get_revoked_token(self, jti):
        query = self._session.query(TokenBlocklist).filter_by(jti=jti).first()
        return query is not None


app = Flask(__name__)
app.config['SECRET_KEY'] = '<insert secret key>'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
jwt = JWTManager(app)


@app.before_request
def protect_bearer_only():
    if not request.headers.get("Authorization"):
        abort(401)


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    role_id = data.get('role_id')

    user_svc = UserService("<insert postgres uri>")
    user_svc.register_user(username, password, email, role_id)
    return {"message": "User registered successfully"}, 201


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get('username')
    data.get('password')

    user_svc = UserService("<insert postgres uri>")
    user = user_svc.authenticate_user(username)
    if not user:
        return {"message": "Invalid credentials"}, 401

    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token), 200


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    user_svc = UserService("<insert postgres uri>")
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200


@jwt.token_in_blocklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService("<insert postgres uri>")
    return user_svc.get_revoked_token(jti)


if __name__ == "__main__":
    app.run()
