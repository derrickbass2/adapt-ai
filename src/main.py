# user_service/main.py

from src.user_repository import UserRepository
from src.user_controller import UserController

def main():
    # Initialize the UserRepository
    user_repository = UserRepository()

    # Initialize the UserController
    user_controller = UserController(user_repository)

    # Call a function from the controller
    print(user_controller.get_users())

if __name__ == "__main__":
    main()