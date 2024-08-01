from pydantic import BaseModel, field_validator, EmailStr
from pydantic_core.core_schema import FieldValidationInfo
import datetime
# 데이터의 도메인을지정해주고
# 이를 검증해준다.
from pydantic import BaseModel

class UserCreate(BaseModel):
    email:str
    password:str
    name:str
    birth_date:datetime.date
    address:str
    
    @field_validator('name', 'password', 'email', 'address')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("빈 값은 허용되지 않습니다.")
        return v
    
    
    
class Token(BaseModel):
    access_token:str
    token_type:str
    name:str