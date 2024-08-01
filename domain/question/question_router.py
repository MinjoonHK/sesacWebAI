from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
from domain.question import question_schema, question_crud
from domain.answer import answer_schema, answer_crud
from starlette import status

from chatbot import Vectorstore, run_query
from datetime import datetime


router = APIRouter(
    prefix="/api/question",
)



@router.post('/chat', status_code=status.HTTP_200_OK)
def chatbot_result(request:question_schema.AskQuestion, db:Session = Depends(get_db)):
     query = request.content
     answer = run_query(Vectorstore, request.content)
     now = datetime.now()
     question = question_crud.insert_question(db=db, _question=request)
     
     created_answer = answer_schema.AnswerResponse(content=answer['response'], create_date=now, question_id=question.id)
     answer_crud.insert_answer(_answer=created_answer, db=db)
     
     return {'answer': answer}
    