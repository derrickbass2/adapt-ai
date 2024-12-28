from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from src.adapt_backend.database_models import Role, User, TokenBlocklist

db = SQLAlchemy()


class UserService:
    def __init__(self):
        """Initialize the UserService with the database session."""
        self._session = db.session

    def create_user(self, username, email, password, role_name="user"):
        """Create a new user with the provided user data."""
        role = self._session.query(Role).filter_by(name=role_name).first()
        if not role:
            role = Role(name=role_name)
            self._session.add(role)
            self._session.commit()

        hashed_password = generate_password_hash(password, method='sha256')
        user = User(username=username, email=email, password=hashed_password, role_id=role.id)

        try:
            self._session.add(user)
            self._session.commit()
            return {"message": "User created successfully", "user_id": user.id}
        except Exception as e:
            print(f"Error occurred while creating user: {e}")
            self._session.rollback()
            return {"message": "Error occurred while creating user", "error": str(e)}

    def login_user(self, username, password):
        """Authenticate and authorize a user with the provided login data."""
        user = self._session.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            return {"message": "User authenticated successfully", "user_id": user.id}
        else:
            return {"message": "Invalid username or password"}

    def get_users(self):
        """Retrieve all users."""
        try:
            users = self._session.query(User).all()
            return [user.to_dict() for user in users]  # Assuming to_dict method exists in User model
        except Exception as e:
            print(f"Error occurred while retrieving users: {e}")
            return {"message": "Error occurred while retrieving users", "error": str(e)}

    def get_user_by_id(self, user_id):
        """Retrieve a user by their ID."""
        try:
            user = self._session.query(User).filter_by(id=user_id).first()
            return user.to_dict() if user else {"message": "User not found"}
        except Exception as e:
            print(f"Error occurred while retrieving user by ID: {e}")
            return {"message": "Error occurred while retrieving user by ID", "error": str(e)}

    def update_user(self, user_id, updated_data):
        """Update a user's information with the provided updated data."""
        try:
            user = self._session.query(User).filter_by(id=user_id).first()
            if not user:
                return {"message": "User not found"}

            for key, value in updated_data.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            self._session.commit()
            return {"message": "User updated successfully", "user_id": user.id}
        except Exception as e:
            print(f"Error occurred while updating user: {e}")
            self._session.rollback()
            return {"message": "Error occurred while updating user", "error": str(e)}

    def delete_user(self, user_id):
        """Delete a user by their ID."""
        try:
            user = self._session.query(User).filter_by(id=user_id).first()
            if not user:
                return {"message": "User not found"}

            self._session.delete(user)
            self._session.commit()
            return {"message": "User deleted successfully"}
        except Exception as e:
            print(f"Error occurred while deleting user: {e}")
            self._session.rollback()
            return {"message": "Error occurred while deleting user", "error": str(e)}

    def revoke_token(self, jti):
        """Revoke a JWT token by adding it to the token blocklist."""
        blocklist = TokenBlocklist(jti=jti)
        try:
            self._session.add(blocklist)
            self._session.commit()
            return {"message": "Token revoked successfully"}
        except Exception as e:
            print(f"Error occurred while revoking token: {e}")
            self._session.rollback()
            return {"message": "Error occurred while revoking token", "error": str(e)}

    def get_revoked_token(self, jti):
        """Check if a JWT token is in the token blocklist."""
        return self._session.query(TokenBlocklist).filter_by(jti=jti).first() is not None
