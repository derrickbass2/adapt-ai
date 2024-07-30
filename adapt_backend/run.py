# run.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.user_repository import UserRepository
from src.user_controller import UserController


# Function to create and return a new session
def get_session(db_uri):
    engine = create_engine(db_uri)
    session = sessionmaker(bind=engine)
    return session()


def main():
    # Database URI
    db_uri = "postgresql://dbuser:dbpassword@localhost:5432/mydatabase"

    # Initialize the Session
    session = get_session(db_uri)

    # Initialize the UserRepository with the session
    UserRepository(session)

    # Initialize the UserController with the UserRepository
    user_controller = UserController()

    # Call a function from the controller
    print(user_controller.get_users())


if __name__ == "__main__":
    main()
