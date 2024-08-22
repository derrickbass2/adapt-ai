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
from sqlalchemy import Column, Integer, String  # Add this line to import the missing modules

Base = declarative_base()

class Role(Base):
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False, index=True)
    users = relationship("User", backref="role")

class User(Base):
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("role.id"), default=1)
    created_at = Column(func.now(), default=None)
    modified_at = Column(func.now(), onupdate=None)

class TokenBlocklist(Base):
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(func.now(), default=None)

class UserService():
    def __init__(self, db_uri):
        engine = create_engine(db_uri)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self._session = Session()

    @staticmethod
    def identity_lookup(payload):
        user_id = payload["sub"]
        return User.query.filter_by(id=user_id).first()

    def register_user(self, username, password, email, role_name):
        user = User(username=username, password=password, email=email, role_name=role_name)
        try:
            self._session.add(user)
            self._session.commit()
        except Exception as e:
            print(f"Error occurred while inserting user {e}")

    def authenticate_user(self, username, password):
        user = User.query.filter_by(username=username).first()
        if user and safe_str_cmp(user.password.encode("utf-8"), password.encode("utf-8")):
            return user

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception as e:
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
    pass

@app.route("/login", methods=["POST"])
def login():
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

if __name__ == "__main__":
    app.run()