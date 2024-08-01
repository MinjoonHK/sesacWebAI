from domain.user.user_schema import UserCreate
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from models import User


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


def get_existing_user(db:Session, user_create:UserCreate):
    user = db.query(User).filter(User.email==user_create.email).first()
    return user


def create_user(db:Session, user_create:UserCreate):
    db_user = User(name=user_create.name,
                   password=pwd_context.hash(user_create.password),
                   email=user_create.email, 
                   birth_date=user_create.birth_date,
                   address=user_create.address)
    db.add(db_user)
    db.commit()
    

def get_user(db:Session, email:str):
    return db.query(User).filter(User.email==email).first()


