from sqlalchemy import create_engine, sqlalchemy
from sqlalchemy.pool import QueuePool
from decouple import config

DATABASE_URL = f"postgresql://{config('DB_USERNAME')}:{config('DB_PASSWORD')}@{config('DB_HOST')}/{config('DB_NAME')}"

engine = create_engine(DATABASE_URL, poolclass=QueuePool)