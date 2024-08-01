from models import Answer
from sqlalchemy.orm import Session
from domain.answer import answer_schema
import datetime

# class Question(Base):
#     __tablename__ = 'question'
    
#     id = Column(Integer, primary_key=True)
#     user_id = Column(Integer, ForeignKey('user.id'))
#     user = relationship('User', backref='questions')
#     content = Column(Text, nullable=False)
#     create_date = Column(DateTime, nullable=False)


# 질문에 의한 답변 정보
# class Answer(Base):
#     __tablename__ = 'answer'
    
#     id = Column(Integer, primary_key=True)
#     content = Column(Text, nullable=False)
#     create_date = Column(DateTime, nullable=False)
#     user_id = Column(Integer, ForeignKey('user.id'))
#     question_id = Column(Integer, ForeignKey('question.id'))
#     user = relationship('User', backref='answer')
#     question = relationship('Question', backref="answer")
    




def insert_answer(db:Session, _answer:answer_schema.AnswerResponse):
    created_Answer = Answer(content = _answer.content, create_date = datetime.datetime.now(), user_id=1, question_id=_answer.question_id)
    db.add(created_Answer)
    db.commit()
    return None
