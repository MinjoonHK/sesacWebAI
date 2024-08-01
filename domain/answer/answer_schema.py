import datetime

from pydantic import BaseModel

class AnswerResponse(BaseModel):
    content:str
    create_date:datetime.datetime
    question_id:int