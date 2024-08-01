from models import Question
from sqlalchemy.orm import Session
from domain.question import question_schema
from datetime import datetime

# class Question(Base):
#     __tablename__ = 'question'
    
#     id = Column(Integer, primary_key=True)
#     user_id = Column(Integer, ForeignKey('user.id'))
#     user = relationship('User', backref='questions')
#     content = Column(Text, nullable=False)
#     create_date = Column(DateTime, nullable=False)


def get_question_list(db:Session):
    question_list = db.query(Question).order_by(Question.create_date.desc()).all()
    return question_list

def insert_question(db:Session, _question:question_schema.AskQuestion):
    new_question = Question(user_id=1, content = _question.content, create_date = datetime.now())
    db.add(new_question)
    db.commit()
    return new_question