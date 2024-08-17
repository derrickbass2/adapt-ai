from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base

# Define metadata
metadata = MetaData()

# Define the base class using the metadata
Base = declarative_base(metadata=metadata)

# Define SQLAlchemy models
class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

# If you have more models, define them here