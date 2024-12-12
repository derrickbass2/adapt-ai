import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError

# Import modular learning system components
import modular_learning_system.spark_engine as spark_engine
from src.app.routes_aa_genome import aa_genome_bp
from src.app.routes_neurotech_network import neurotech_network_bp
from src.app.routes_spark_engine import spark_engine_bp

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Set up the Flask configuration using environment variables
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DB_URI")  # Use DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False  # Optional, to suppress warnings
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
app.config["JWT_BLACKLIST_ENABLED"] = True
app.config["JWT_BLACKLIST_TOKEN_CHECKS"] = ["access", "refresh"]
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

# Initialize SQLAlchemy and JWTManager
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Initialize the Spark Engine
spark_engine_instance = spark_engine.SparkEngine()

# Register the blueprints with the main application
app.register_blueprint(aa_genome_bp, url_prefix='/api/aa-genome')
app.register_blueprint(neurotech_network_bp, url_prefix='/api/neurotech-network')
app.register_blueprint(spark_engine_bp, url_prefix='/api/spark-engine')


# Database models
class Role(db.Model):
    __tablename__ = "roles"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False, index=True)
    users = db.relationship("User", backref="role")


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey("roles.id"), default=1)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    modified_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


class TokenBlocklist(db.Model):
    __tablename__ = "token_blocklist"
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(128), nullable=False, index=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# Service class for handling user operations
class UserService:
    def __init__(self, db_session):
        self._session = db_session

    def register_user(self, username, email, password, role_name="user"):
        role = self._session.query(Role).filter_by(name=role_name).first()
        if not role:
            role = Role(name=role_name)
            self._session.add(role)
            self._session.commit()

        user = User(username=username, email=email, password=password, role_id=role.id)
        try:
            self._session.add(user)
            self._session.commit()
        except SQLAlchemyError as e:
            print(f"Error occurred while inserting user: {e}")
            self._session.rollback()

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except SQLAlchemyError as e:
            print(f"Error occurred while inserting blacklisted token: {e}")
            self._session.rollback()

    def get_revoked_token(self, jti):
        return self._session.query(TokenBlocklist).filter_by(jti=jti).first() is not None


# JWT token blocklist loader
@jwt.token_in_blocklist_loader
def check_if_token_in_blocklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService(db.session)
    return user_svc.get_revoked_token(jti)


# Flask routes
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    role_name = data.get("role", "user")

    user_svc = UserService(db.session)
    user_svc.register_user(username, email, password, role_name)

    return jsonify({"message": "User registered successfully"}), 201


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    user_svc = UserService(db.session)
    user_svc.revoke_token(jti)
    return jsonify({"message": "Successfully logged out"}), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    # Get the JSON data sent from the client
    data = request.json

    # Perform prediction using the Spark Engine
    result = spark_engine_instance.predict(data)

    # Return a JSON response
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)


def create_app():
    return None
