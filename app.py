import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import func

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Set up the Flask configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DB_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional, to suppress warnings
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']

db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Define the database models
class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False, index=True)
    users = db.relationship("User", backref="role")

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'), default=1)
    created_at = db.Column(db.DateTime, server_default=func.now())
    modified_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class TokenBlocklist(db.Model):
    __tablename__ = 'token_blocklist'
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(128), nullable=False, index=True)
    created_at = db.Column(db.DateTime, server_default=func.now())

# Service class to handle user operations
class UserService:
    def __init__(self, db_session):
        self._session = db_session

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
        except Exception as e:
            print(f"Error occurred while inserting user: {e}")
            self._session.rollback()

    def authenticate_user(self, username):
        return self._session.query(User).filter_by(username=username).first()

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception as e:
            print(f"Error occurred while inserting blacklisted token: {e}")
            self._session.rollback()

    def get_revoked_token(self, jti):
        return self._session.query(TokenBlocklist).filter_by(jti=jti).first() is not None

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
    user_svc = UserService(db.session)
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200

@jwt.token_in_blocklist_loader
def check_if_token_in_blocklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService(db.session)
    return user_svc.get_revoked_token(jti)

if __name__ == "__main__":
    app.run(debug=True)
