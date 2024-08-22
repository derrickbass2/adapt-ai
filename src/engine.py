from decouple import config
from sqlalchemy import create_engine, pool

DATABASE_URL = f"postgresql://{config('DB_USER')}:{config('DB_PASSWORD')}@{config('DB_HOST')}:{config('DB_PORT')}/{config('DB_NAME')}"

engine = create_engine(DATABASE_URL, poolclass=pool.QueuePool)
