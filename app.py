from datetime import timedelta

from flask import Flask, abort
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class Role(Base):
    __tablename__ = 'roles'  # Specify the table name

    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False, index=True)
    users = relationship("User", backref="role")


class User(Base):
    __tablename__ = 'users'  # Specify the table name

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    role_id = Column(Integer, ForeignKey('roles.id'), default=1)  # Correct ForeignKey usage
    created_at = Column(func.now(), default=None)
    modified_at = Column(func.now())


class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'  # Specify the table name

    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(func.now(), default=None)


class UserService:
    def __init__(self, db_uri):
        engine = create_engine(db_uri)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        self._session = Session()

    @staticmethod
    def identity_lookup(payload, self=None):
        user_id = payload["sub"]
        return self._session.query(User).filter_by(id=user_id).first()

    def register_user(self, username, password, email, role_name):
        role = self._session.query(Role).filter_by(name=role_name).first()
        if not role:
            role = Role(name=role_name)
            self._session.add(role)
            self._session.commit()

        user = User(username=username, password=password, email=email)
        try:
            self._session.add(user)
            self._session.commit()
        except Exception:
            print("Error occurred while inserting user: {e}")
            self._session.rollback()

    def authenticate_user(self, username):
        user = self._session.query(User).filter_by(username=username).first()
        if user:
            return user

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception:
            print("Error occurred while inserting blacklisted token: {e}")
            self._session.rollback()

    def get_revoked_token(self, jti):
        query = self._session.query(TokenBlocklist).filter_by(jti=jti).first()
        return query is not None


app = Flask(__name__)
app.config['SECRET_KEY'] = b'Pc\xe2\xe5a@\x96\xa6\xd7\xaa2\xfb\xde\xd1U!\x07\x10\x00\xfdDlS\x8b'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
jwt = JWTManager(app)


@app.before_request
def protect_bearer_only(authorized=None):
    if not authorized:
        abort(401)


@app.route("/register", methods=["POST"])
def register():
    # Placeholder for registration logic
    return {"message": "Registration endpoint"}, 200


@app.route("/login", methods=["POST"])
def login():
    # Placeholder for login logic
    return {"message": "Login endpoint"}, 200


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    user_svc = UserService("postgresql://dbuser:mypassword@localhost:5432/mydatabase")
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200


@jwt.token_in_blacklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService("postgresql://dbuser:mypassword@localhost:5432/mydatabase")
    return user_svc.get_revoked_token(jti)


if __name__ == "__main__":
    app.run()


def create_app():
    return app
