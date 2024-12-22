import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from adaptai.routes import UserController
from adaptai.user_repository import UserRepository


def main():
    try:
        # Fetch database URI from an environment variable
        db_uri = os.getenv("DATABASE_URI")
        if not db_uri:
            raise ValueError("DATABASE_URI environment variable is not set")

        # Create a database engine
        engine = create_engine(db_uri)

        # Create a configured "session" class
        Session = sessionmaker(bind=engine)

        # Create a session instance
        with Session() as session:
            # Initialize the UserRepository with the session
            user_repository = UserRepository(session)

            # Initialize the UserController with the UserRepository
            user_controller = UserController(user_repository)

            # Call a function from the controller
            print(user_controller.get_users())

    except Exception as e:
        # Handle unexpected errors
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
