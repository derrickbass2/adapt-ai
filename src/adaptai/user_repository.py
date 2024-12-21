from contextlib import contextmanager

from sqlalchemy import Table, MetaData, select, update, delete, insert
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Session

from .schemas import UserSchema


class UserRepository:
    def __init__(self, session: Session):
        """
        Initialize the UserRepository with a SQLAlchemy session.

        Args:
            session (Session): SQLAlchemy database session.
        """
        self.session = session
        self.metadata = MetaData()
        self.users_table = Table("users", self.metadata, autoload_with=session.bind)

    @contextmanager
    def managed_transaction(self):
        """
        Helper function to automatically manage database transactions.
        Rolls back the session on exception and commits on success.
        """
        try:
            yield self.session
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def create_user(self, user_schema: UserSchema) -> UserSchema:
        """
        Create a new user in the database.

        Args:
            user_schema (UserSchema): User data.

        Returns:
            UserSchema: The newly created user data.
        """
        with self.managed_transaction():
            user_data = user_schema.model_dump()
            stmt = insert(self.users_table).values(**user_data).returning(*self.users_table.columns)
            result = self.session.execute(stmt).fetchone()
            if result:
                return UserSchema(**result._asdict())
            else:
                raise IntegrityError("Failed to insert user", orig=None, params=user_data)

    def get_user_by_email(self, email: str) -> UserSchema:
        """
        Retrieve a user by email.

        Args:
            email (str): The user's email.

        Returns:
            UserSchema: The user record.
        """
        try:
            stmt = select(self.users_table).where(self.users_table.c.email == email)
            result = self.session.execute(stmt).fetchone()
            if result:
                return UserSchema(**result._asdict())
            else:
                raise NoResultFound(f"User with email {email} not found")
        except NoResultFound as e:
            raise e

    def get_users(self) -> list[UserSchema]:
        """
        Fetch all users from the database.

        Returns:
            list[UserSchema]: A list of all user records.
        """
        stmt = select(self.users_table)
        result = self.session.execute(stmt)
        return [UserSchema(**row._asdict()) for row in result.fetchall()]

    def update_user(self, id: int, user_schema: UserSchema) -> UserSchema:
        """
        Update an existing user by ID.

        Args:
            id (int): The user's ID.
            user_schema (UserSchema): User data to update.

        Returns:
            UserSchema: The updated user data.
        """
        with self.managed_transaction():
            user_data = user_schema.model_dump()
            stmt = (
                update(self.users_table)
                .where(self.users_table.c.id == id)
                .values(**user_data)
                .returning(*self.users_table.columns)
            )
            result = self.session.execute(stmt).fetchone()
            if result:
                return UserSchema(**result._asdict())
            else:
                raise NoResultFound(f"User with id {id} not found")

    def delete_user(self, id: int) -> None:
        """
        Delete a user by ID.

        Args:
            id (int): The user's ID.

        Returns:
            None
        """
        with self.managed_transaction():
            stmt = delete(self.users_table).where(self.users_table.c.id == id)
            result = self.session.execute(stmt)
            if result.rowcount == 0:
                raise NoResultFound(f"User with id {id} not found")
