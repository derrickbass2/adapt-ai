
from sqlalchemy.orm import Session

from ..database import Base
from ..schemas import UserSchema

class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_user(self, user_schema: UserSchema):
        user_obj = Base.metadata.tables["users"].insert().values(**user_schema.dict())
        self.session.execute(Base.metadata.tables["users"].insert(), user_obj)
        return user_schema

    def get_user_by_email(self, email: str):
        query = self.session.query(Base.metadata.tables["users"]).filter_by(email=email)
        user = query.first()
        return user

    def get_users(self):
        query = self.session.query(Base.metadata.tables["users"])
        users = query.all()
        return users

    def update_user(self, id: int, user_schema: UserSchema):
        user_obj = {column.name: user_schema.dict()[column.name] for column in Base.metadata.tables["users"].columns}
        self.session.query(Base.metadata.tables["users"]).filter_by(id=id).update(user_obj)
        return user_schema

    def delete_user(self, id: int):
        self.session.query(Base.metadata.tables["users"]).filter_by(id=id).delete()