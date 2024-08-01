from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from database import SessionLocal, get_db
from domain.user import user_schema, user_crud

from datetime import timedelta, datetime
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from domain.user.user_crud import pwd_context


ACCESS_TOKEN_EXPIRE_MINUTES = 60*24
SECRET_KEY = "2d837b1b41321436a9ed738aa91b6c0aedfe58c67f8173aa27f245072d6ec11c"
ALGORITHM = "HS256"

router = APIRouter(
    prefix="/api/user",
)

@router.post('/create', status_code=status.HTTP_201_CREATED)
def user_create(_user_create:user_schema.UserCreate, db:Session=Depends(get_db)):
    user = user_crud.get_existing_user(db=db, user_create=_user_create)
    if user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="이미 존재하는 사용자입니다.")
    user_crud.create_user(db = db, user_create=_user_create)
   
    return {'message':"회원가입이 성공적으로 완료되었습니다", 'status':"success"}
    
    
    
    
    
@router.post('/login', response_model=user_schema.Token, status_code=status.HTTP_202_ACCEPTED)
def login_for_access_token(form_data:OAuth2PasswordRequestForm = Depends(), db:Session=Depends(get_db)):
    user = user_crud.get_user(db, form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                            detail="이메일 혹은 비밀번호가 잘못되었습니다.",
                            headers={'WWW-Authenticate':'Bearer'},)
        
    data = {
        'sub':user.email,
        'exp':datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
        
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
        
    return {
        'message':"로그인 완료",
        'access_token':access_token,
        'token_type':"bearer",
        'name':user.email
    }
    