import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
from sqlalchemy import Column, Integer, String, DateTime, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

load_dotenv()

Base = declarative_base()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DB_URI')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']

# Initialize Flask-Migrate
from flask_migrate import Migrate

migrate = Migrate(app, Base)

jwt = JWTManager(app)


class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False, index=True)
    users = relationship("User", backref="role")


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    role_id = Column(Integer, ForeignKey('roles.id'), default=1)
    created_at = Column(DateTime, server_default=func.now())
    modified_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())


class UserService:
    def __init__(self, db_uri):
        engine = create_engine(db_uri)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self._session = Session()

    def identity_lookup(self, payload):
        user_id = payload["sub"]
        return self._session.query(User).filter_by(id=user_id).first()

    def register_user(self, username, email, password, role_name='user'):
        role = self._session.query(Role).filter_by(name=role_name).first()
        if not role:
            role = Role(name=role_name)
            self._session.add(role)
            self._session.commit()

        user = User(username=username, email=email, password=password, role_id=role.id)
        try:
            self._session.add(user)
            self._session.commit()
        except Exception:
            print("Error occurred while inserting user: {e}")
            self._session.rollback()

    def authenticate_user(self, username):
        return self._session.query(User).filter_by(username=username).first()

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception:
            print("Error occurred while inserting blacklisted token: {e}")
            self._session.rollback()

    def get_revoked_token(self, jti):
        return self._session.query(TokenBlocklist).filter_by(jti=jti).first() is not None


@app.before_request
def protect_bearer_only():
    os.abort()


@app.route("/register", methods=["POST"])
def register():
    return {"message": "Registration endpoint"}, 200


@app.route("/login", methods=["POST"])
def login():
    return {"message": "Login endpoint"}, 200


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    user_svc = UserService(os.getenv('DB_URI'))
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200


@jwt.token_in_blocklist_loader
def check_if_token_in_blocklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService(os.getenv('DB_URI'))
    return user_svc.get_revoked_token(jti)


if __name__ == "__main__":
    app.run()
