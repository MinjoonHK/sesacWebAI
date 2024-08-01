from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Date
from sqlalchemy.orm import relationship

from database import Base

# 유저가 보내는 질문 정보
class Question(Base):
    __tablename__ = 'question'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'))
    user = relationship('User', backref='questions')
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    
    # answers 백레퍼런스를 추가합니다.
    answers = relationship('Answer', back_populates='question')
    
# 유저의 정보
class User(Base):
    __tablename__ = 'user'
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True,nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, unique=True, nullable=False)
    birth_date = Column(Date, nullable=False)
    address = Column(String, nullable=False)
    

# 질문에 의한 답변 정보
class Answer(Base):
    __tablename__ = 'answer'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey('user.id'))
    question_id = Column(Integer, ForeignKey('question.id'))
    user = relationship('User', backref='answers')
    question = relationship('Question', back_populates="answers")


# class Post(Base):
#     __tablename__ = 'post'

#     id = Column(Integer, primary_key=True)
#     subject = Column(String, nullable=False)
#     content = Column(Text, nullable=False)
#     create_date = Column(DateTime, nullable=False)
#     user_id = Column(Integer, ForeignKey('user.id'))
    
#     user = relationship('User', backref='posts')
#     comments = relationship('Comment', back_populates='post')


# class Comment(Base):
#     __tablename__ = 'comment'

#     id = Column(Integer, primary_key=True)
#     content = Column(Text, nullable=False)
#     create_date = Column(DateTime, nullable=False)
#     user_id = Column(Integer, ForeignKey('user.id'))
#     post_id = Column(Integer, ForeignKey('post.id'))

#     user = relationship('User', backref='comments')
#     post = relationship('Post', back_populates='comments')
