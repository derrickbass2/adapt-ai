# /Users/derrickbass/Public/adaptai/src/adapt_backend/models.py

from datetime import datetime

from sqlalchemy import Column, Integer, String, MetaData, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Define metadata
metadata = MetaData()

# Define the base class using the metadata
Base = declarative_base(metadata=metadata)


# Define SQLAlchemy models
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)

    role = relationship('Role', back_populates='users')

    def to_dict(self):
        """Convert User model to a dictionary for easy serialization."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "role_id": self.role_id
        }


class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    users = relationship('User', back_populates='role')


class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'

    id = Column(Integer, primary_key=True)
    jti = Column(String(36), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# If you have more models, define them here
class LRModel:
    pass


class RFModel:
    pass


class SVMModel:
    pass


def db():
    return None
