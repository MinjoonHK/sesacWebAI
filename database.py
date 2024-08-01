#database.py 파일은 데이터베이스와 관련된 설정을 하는 파일이다.
# 이 파일에는 데이터베이스를 사용하기 위한 변수, 함수등을 정의하고
# 접속할 데이터베이스의 주소와 사용자, 비밀번호등을 관리한다.

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = 'sqlite:///./myapi.db'

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

#데이터 베이스 접속
SessionLocal = sessionmaker(autocommit=False, autoflush=False,bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()