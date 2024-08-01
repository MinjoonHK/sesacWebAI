import datetime
# 데이터의 도메인을지정해주고
# 이를 검증해준다.
from pydantic import BaseModel, field_validator

class Question(BaseModel):
    id:int
    subject:str | None
    content:str
    create_date:datetime.datetime
    
    
class AskQuestion(BaseModel):
    content:str
    
    
    @field_validator('content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("빈 값은 허용되지 않습니다.")
        if not (len(v) > 1 and len(v) < 100):
            raise ValueError("1글자 이상 100글자 이하로 작성해주세요.")
        return v
