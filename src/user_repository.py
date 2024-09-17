from sqlalchemy import Table, MetaData
from sqlalchemy.orm import Session

from .schemas import UserSchema


class UserRepository:
    def __init__(self, session: Session):
        self.session = session
        self.metadata = MetaData(bind=session.bind)
        self.users_table = Table('users', self.metadata, autoload_with=session.bind)

    def create_user(self, user_schema: UserSchema):
        user_obj = user_schema.model_dump()
        self.session.execute(self.users_table.insert().values(**user_obj))
        self.session.commit()
        return user_schema

    def get_user_by_email(self, email: str):
        query = self.session.query(self.users_table).filter_by(email=email)
        user = query.first()
        return user

    def get_users(self):
        query = self.session.query(self.users_table)
        users = query.all()
        return users

    def update_user(self, id: int, user_schema: UserSchema):
        user_obj = {column.name: user_schema.model_dump().get(column.name) for column in self.users_table.columns}
        self.session.query(self.users_table).filter_by(id=id).update(user_obj)
        self.session.commit()
        return user_schema

    def delete_user(self, id: int):
        self.session.query(self.users_table).filter_by(id=id).delete()
        self.session.commit()
