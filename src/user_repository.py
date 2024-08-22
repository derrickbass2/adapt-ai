from sqlalchemy import MetaData
from sqlalchemy.orm import Session
from .schemas import UserSchema


class UserRepository:
    def __init__(self, session: Session):
        self.session = session
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.session.bind)
        self.users_table = self.metadata.tables["users"]

    def create_user(self, user_schema: UserSchema):
        user_dict = user_schema.model_dump()
        self.session.execute(self.users_table.insert().values(**user_dict))
        self.session.commit()  # Ensure you commit the transaction
        return user_schema

    def get_user_by_email(self, email: str):
        query = self.session.query(self.users_table).filter_by(email=email)
        user = query.first()
        return user

    def get_users(self):
        query = self.session.query(self.users_table)
        users = query.all()
        return users

    def update_user(self, user_id: int, user_schema: UserSchema) -> object:
        user_dict = user_schema.model_dump()
        # Remove 'id' from the update dictionary as it should not be updated
        if 'id' in user_dict:
            user_dict.pop('id')
        self.session.query(self.users_table).filter_by(id=user_id).update(user_dict)
        self.session.commit()  # Ensure you commit the transaction
        return user_schema

    def delete_user(self, user_id: int) -> bool:
        result = self.session.query(self.users_table).filter_by(id=user_id).delete()
        self.session.commit()  # Ensure you commit the transaction
        return result > 0  # Return True if a row was deleted, otherwise False
