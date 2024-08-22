<<<<<<< HEAD
from decouple import config
from sqlalchemy import create_engine, pool

DATABASE_URL = f"postgresql://{config('DB_USER')}:{config('DB_PASSWORD')}@{config('DB_HOST')}:{config('DB_PORT')}/{config('DB_NAME')}"
=======
from sqlalchemy import create_engine, pool
from decouple import config
import sqlalchemy

DATABASE_URL = f"postgresql://{config('DB_USERNAME')}:{config('DB_PASSWORD')}@{config('DB_HOST')}/{config('DB_NAME')}"
>>>>>>> 721ae5e8 (Delete unnecessary files from virtual environment)

engine = create_engine(DATABASE_URL, poolclass=pool.QueuePool)
