from adaptai.user_controller import UserController
from adaptai.user_repository import UserRepository


def main():
    # Initialize the UserRepository
    user_repository = UserRepository()

    # Initialize the UserController
    user_controller = UserController(user_repository)

    # Call a function from the controller
    print(user_controller.get_users())


if __name__ == "__main__":
    main()
