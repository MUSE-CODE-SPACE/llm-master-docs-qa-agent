"""FastAPI: expose the agent as a web API. / 에이전트를 웹 API로 노출."""
from fastapi import FastAPI

from qa.agent import answer_question
from qa.models import AnswerResponse, AskRequest

app = FastAPI(title="Docs QA Agent")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest):
    return answer_question(req.question)
