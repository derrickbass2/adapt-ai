from src.user_repository import UserRepository
from src.user_controller import UserController
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def main():
    # Create a database engine
    db_uri = "postgresql://dbuser:dbpassword@localhost:5432/mydatabase"
    engine = create_engine(db_uri)

    # Create a configured "session" class
    session = sessionmaker(bind=engine)

    # Create a session instance
    session = session()

    # Initialize the UserRepository with the session
    UserRepository(session)

    # Initialize the UserController with the UserRepository
    user_controller = UserController()

    # Call a function from the controller
    print(user_controller.get_users())


if __name__ == "__main__":
    main()