from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from domain.question import question_router
from domain.user import user_router
import uvicorn

app = FastAPI()


#CORS 오류를 해결하기 위해 이렇게 할 수 있다.
# 프론트엔드의 URL을 통해서 할 수 있을듯!
origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # 'allow_credential'에서 's'를 추가했습니다.
    allow_methods=['*'],
    allow_headers=["*"]
)

app.include_router(question_router.router)
app.include_router(user_router.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=80,
    )