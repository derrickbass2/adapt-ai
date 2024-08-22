from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_URL = "postgresql://{config('DB_USERNAME')}:{config('DB_PASSWORD')}@{config('DB_HOST')}/{config('DB_NAME')}"

engine = create_engine(DATABASE_URL, poolclass=QueuePool)
