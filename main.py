from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from adaptai.user_controller import UserController
from adaptai.user_repository import UserRepository


def main():
    # Create a database engine
    db_uri = "postgresql://dbuser:dbpassword@localhost:5432/mydatabase"
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


if __name__ == "__main__":
    main()
