<<<<<<< HEAD
import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DateTime
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
import flask
from werkzeug.security import generate_password_hash, check_password_hash
=======
from datetime import timedelta
from flask import Flask, Blueprint, jsonify, request, session
from flask_jwt_extended import JWTManager, jwt_required, decode_token, create_access_token, \
    get_raw_jwt, get_jti
from authlib.integrations.flask_oauth2 import ResourceProtector, AuthorizationServer, \
    BearerTokenValidator
from authlib.specs.rfc6749 import GrantType
from werkzeug.security import safe_str_cmp
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from flask import abort
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String  # Add this line to import the missing modules
from sqlalchemy.orm import sessionmaker, relationship  # Add this line to import the missing module
>>>>>>> cleanup/duplicate-removal

Base = declarative_base()

class Role(Base):
<<<<<<< HEAD
    __tablename__ = 'roles'
=======
>>>>>>> cleanup/duplicate-removal
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False, index=True)
    users = relationship("User", backref="role")

class User(Base):
<<<<<<< HEAD
    __tablename__ = 'users'
=======
>>>>>>> cleanup/duplicate-removal
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
<<<<<<< HEAD
    role_id = Column(Integer, ForeignKey("roles.id"), default=1)
    created_at = Column(DateTime, server_default=func.now())
    modified_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())

class UserService:
=======
    role_id = Column(Integer, ForeignKey("role.id"), default=1)
    created_at = Column(func.now(), default=None)
    modified_at = Column(func.now(), onupdate=None)

class TokenBlocklist(Base):
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(func.now(), default=None)

class UserService():
>>>>>>> cleanup/duplicate-removal
    def __init__(self, db_uri):
        engine = create_engine(db_uri)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self._session = Session()

<<<<<<< HEAD
    def identity_lookup(self, payload):
        user_id = payload["sub"]
        return self._session.query(User).filter_by(id=user_id).first()

    def register_user(self, username, password, email):
        hashed_password = generate_password_hash(password)
        user = User(username=username, password=hashed_password, email=email)
=======
    @staticmethod
    def identity_lookup(payload):
        user_id = payload["sub"]
        return User.query.filter_by(id=user_id).first()

    def register_user(self, username, password, email, role_name):
        user = User(username=username, password=password, email=email, role_name=role_name)
>>>>>>> cleanup/duplicate-removal
        try:
            self._session.add(user)
            self._session.commit()
        except Exception as e:
<<<<<<< HEAD
            self._session.rollback()
            print(f"Error occurred while inserting user: {e}")

    def authenticate_user(self, username, password):
        user = self._session.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
=======
            print(f"Error occurred while inserting user {e}")

    def authenticate_user(self, username, password):
        user = User.query.filter_by(username=username).first()
        if user and safe_str_cmp(user.password.encode("utf-8"), password.encode("utf-8")):
>>>>>>> cleanup/duplicate-removal
            return user

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception as e:
<<<<<<< HEAD
            self._session.rollback()
            print(f"Error occurred while inserting blacklisted token: {e}")

    def get_revoked_token(self, jti):
        query = self._session.query(TokenBlocklist).filter_by(jti=jti).first()
        return query is not None

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = b'Pc\xe2\xe5a@\x96\xa6\xd7\xaa2\xfb\xde\xd1U!\x07\x10\x00\xfdDlS\x8b'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']

class JWTManager:
    @staticmethod
    def token_in_blocklist_loader(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

jwt = JWTManager()

@app.before_request
def protect_bearer_only():
    from authlib.oauth1 import ResourceProtector
    protector = ResourceProtector()
    authorized = protector.check_authorization_header(
        flask.request.headers.get("Authorization", "")
    )
    if not authorized:
        flask.abort(401)

@app.route("/register", methods=["POST"])
def register():
    data = flask.request.get_json()
    user_svc = UserService("postgresql://dbuser:mypassword@localhost:5432/mydatabase")
    user_svc.register_user(data['username'], data['password'], data['email'])
    return {"message": "User registered successfully"}, 200

def create_access_token():
=======
            print(f"Error occurred while inserting blacklisted token {e}")

    def get_revoked_token(self, jti):
        query = TokenBlocklist.query.filter_by(jti=jti).first()
        if query is None:
            return False
        else:
            return True

app = Flask(__name__)
app.config['SECRET_KEY'] = '<insert secret key>'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
jwt = JWTManager(app)

@app.before_request
def protect_bearer_only():
    protector = ResourceProtector()
    authorized = protector.check_AuthorizationHeader(
        request.headers.get("Authorization", "")
    )
    if not authorized:
        abort(401)

@app.route("/register", methods=["POST"])
def register():
>>>>>>> cleanup/duplicate-removal
    pass

@app.route("/login", methods=["POST"])
def login():
<<<<<<< HEAD
    data = flask.request.get_json()
    user_svc = UserService("postgresql://dbuser:mypassword@localhost:5432/mydatabase")
    user = user_svc.authenticate_user(data['username'], data['password'])
    if user:
        access_token = create_access_token()  # Placeholder for token creation logic
        return {"access_token": access_token}, 200
    else:
        return {"message": "Invalid credentials"}, 401

def get_raw_jwt():
    pass

@app.route("/logout", methods=["POST"])
def logout():
    jti = get_raw_jwt()["jti"]
    user_svc = UserService("postgresql://dbuser:mypassword@localhost:5432/mydatabase")
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200

@jwt.token_in_blocklist_loader
def check_if_token_in_blocklist(jwt_payload):
    jti = jwt_payload["jti"]
    user_svc = UserService("postgresql://dbuser:mypassword@localhost:5432/mydatabase")
    return user_svc.get_revoked_token(jti)
=======
    pass

@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_raw_jwt()["jti"]
    user_svc = UserService("<insert postgres uri>")
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200

@jwt.token_in_blacklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService("<insert postgres uri>")
    if user_svc.get_revoked_token(jti):
        return True
    return False
>>>>>>> cleanup/duplicate-removal

if __name__ == "__main__":
    app.run()