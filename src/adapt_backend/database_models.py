from datetime import datetime

from sqlalchemy import Column, Integer, String, MetaData, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates

# Define metadata
metadata = MetaData()

# Define the base class using the metadata
Base = declarative_base(metadata=metadata)


# User Model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    jti = Column(String(36), nullable=False, index=True)
    password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id', ondelete='CASCADE'), nullable=False)
    role = relationship('Role', back_populates='users')

    def __repr__(self):
        """Provide a string representation for debugging."""
        return (f"<User(id={self.id}, username={self.username}, "
                f"email={self.email}, role_id={self.role_id})>")

    def to_dict(self):
        """Convert User model to a dictionary for easy serialization."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "role_id": self.role_id
        }

    @validates('email')
    def validate_email(self, email):
        """Validate the format of the email address."""
        if '@' not in email or '.' not in email:
            raise ValueError("Invalid email address format.")
        return email

    @validates('username')
    def validate_username(self, username):
        """Validate username length and format."""
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        return username


# Role Model
class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    users = relationship('User', back_populates='role', cascade='all, delete-orphan')

    def __repr__(self):
        """Provide a string representation for debugging."""
        return f"<Role(id={self.id}, name={self.name})>"


# Token Blocklist Model
class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'

    id = Column(Integer, primary_key=True)
    jti = Column(String(36), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        """Provide a string representation for debugging."""
        return f"<TokenBlocklist(id={self.id}, jti={self.jti}, created_at={self.created_at})>"


# Abstract Model Class for Machine Learning Models
class BaseModel:
    """
    Base class for machine learning models to define shared functionality.
    """

    def __init__(self):
        self.model = None

    def save(self, filepath: str):
        """Save the trained model to a file."""
        raise NotImplementedError("The save() method should be implemented by subclasses.")

    def load(self, filepath: str):
        """Load a previously trained model."""
        raise NotImplementedError("The load() method should be implemented by subclasses.")


# Logistic Regression Model Class
class LRModel(BaseModel):
    """Logistic Regression Model with save/load functionality."""

    def save(self, filepath: str):
        print(f"Saving Logistic Regression model to {filepath}...")
        # Add logic to serialize and save the model
        pass

    def load(self, filepath: str):
        print(f"Loading Logistic Regression model from {filepath}...")
        # Add logic to deserialize and load the model
        pass


# Random Forest Model Class
class RFModel(BaseModel):
    """Random Forest Model with save/load functionality."""

    def save(self, filepath: str):
        print(f"Saving Random Forest model to {filepath}...")
        # Add logic to serialize and save the model
        pass

    def load(self, filepath: str):
        print(f"Loading Random Forest model from {filepath}...")
        # Add logic to deserialize and load the model
        pass


# Support Vector Machine Model Class
class SVMModel(BaseModel):
    """Support Vector Machine Model with save/load functionality."""

    def save(self, filepath: str):
        print(f"Saving SVM model to {filepath}...")
        # Add logic to serialize and save the model
        pass

    def load(self, filepath: str):
        print(f"Loading SVM model from {filepath}...")
        # Add logic to deserialize and load the model
        pass
