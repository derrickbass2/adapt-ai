from sqlalchemy.orm import scoped_session, sessionmaker
from database.engine import engine

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

session = scoped_session(SessionLocal)