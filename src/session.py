from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import os

# Define the base class for SQLAlchemy models
Base = declarative_base()


# Create engine and session maker
def get_engine():
    db_uri = os.getenv('DATABASE_URI')  # Ensure DATABASE_URI is set in your environment
    return create_engine(db_uri)


def get_session():
    engine = get_engine()
    session = scoped_session(sessionmaker(bind=engine))
    return session


def query():
    return None
