import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, DateTime
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
import flask
from werkzeug.security import generate_password_hash, check_password_hash

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

    def register_user(self, username, password, email):
        hashed_password = generate_password_hash(password)
        user = User(username=username, password=hashed_password, email=email)
        try:
            self._session.add(user)
            self._session.commit()
        except Exception as e:
            self._session.rollback()
            print(f"Error occurred while inserting user: {e}")

    def authenticate_user(self, username, password):
        user = self._session.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            return user

    def revoke_token(self, jti):
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
        except Exception as e:
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
    pass

@app.route("/login", methods=["POST"])
def login():
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

if __name__ == "__main__":
    app.run()