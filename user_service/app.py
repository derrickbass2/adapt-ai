from datetime import timedelta
from flask import Flask, request, abort
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from werkzeug.security import safe_str_cmp

# Create a base class for SQLAlchemy models
Base = declarative_base()


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
    role_id = Column(Integer, ForeignKey("roles.id"), default=1)
    created_at = Column(func.now(), default=func.now())
    modified_at = Column(func.now(), onupdate=func.now())


class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'
    id = Column(Integer, primary_key=True)
    jti = Column(String(128), nullable=False, index=True)
    created_at = Column(func.now(), default=func.now())


class UserService:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._session = self.Session()

    def identity_lookup(self, payload):
        user_id = payload["sub"]
        return self._session.query(User).filter_by(id=user_id).first()

    def register_user(self, username, password, email, role_id):
        user = User(username=username, password=password, email=email, role_id=role_id)
        try:
            self._session.add(user)
            self._session.commit()
        except Exception as e:
            print("Error occurred while inserting user: {e}")

    def authenticate_user(self, username, password):
        user = self._session.query(User).filter_by(username=username).first()
        if user and safe_str_cmp(user.password.encode("utf-8"), password.encode("utf-8")):
            return user
        return None

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception as e:
            print("Error occurred while inserting blacklisted token: {e}")

    def get_revoked_token(self, jti):
        query = self._session.query(TokenBlocklist).filter_by(jti=jti).first()
        return query is not None


app = Flask(__name__)
app.config['SECRET_KEY'] = b'Pc\xc2\xe5a@\x96\xa6\xd7\xaa2\xfb\xde\xd1U!\x07\x10\x00\xfdDlS\x8b'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)

app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
jwt = JWTManager(app)


@app.before_request
def protect_bearer_only():
    # ResourceProtector should be used with OAuth 1.0a
    # The below line uses OAuth 1.0a which may not be appropriate here
    # Ensure you are using OAuth 1.0a correctly or use OAuth 2.0 if needed
    from authlib.oauth1 import ResourceProtector
    protector = ResourceProtector()
    authorized = protector.check_authorization_header(
        request.headers.get("Authorization", "")
    )
    if not authorized:
        abort(401)


@app.route("/register", methods=["POST"])
def register():
    # Implement user registration logic here
    pass


@app.route("/login", methods=["POST"])
def login():
    # Implement user login logic here
    pass


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    user_svc = UserService("<insert postgres uri>")
    user_svc.revoke_token(jti)
    return {"message": "Successfully logged out"}, 200


@jwt.token_in_blacklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token["jti"]
    user_svc = UserService("<insert postgres uri>")
    return user_svc.get_revoked_token(jti)


if __name__ == "__main__":
    app.run()
