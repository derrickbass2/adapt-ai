from pydantic import BaseModel


class UserSchema(BaseModel):
    id: int
    username: str
    email: str

    # Add other fields if needed

    class Config:
        orm_mode = True

    def model_dump(self):
        pass
