<!-- 🌏 English + 한국어 (English first) -->
# Docs QA Agent — RAG from scratch / 문서 QA 에이전트 — 래그 직접 만들기

> **Find → Read → Answer (with citations).** A working Retrieval-Augmented Generation (RAG) agent: index your docs, ask in natural language, get an answer grounded in the sources with citations.
> **찾아서 → 읽고 → 답한다 (인용과 함께).** 문서를 인덱싱하고 자연어로 물으면, 출처에 근거한 답을 인용과 함께 돌려주는 동작하는 래그(RAG) 에이전트입니다.

> 📱 Hands-on code for the **[LLM Master](https://apps.apple.com/app/id6769785318)** course — Chapter "실전 프로젝트 · Project 1: 문서 QA Agent".
> 학습 앱 **LLM Master** 의 실습 코드 — "실전 프로젝트 1: 문서 QA Agent" 챕터.

```
question ──▶ embed ──▶ Chroma search (top-k) ──▶ context ──▶ LLM ──▶ {answer, citations}
질문 ──▶ 임베딩 ──▶ 크로마 검색 ──▶ 컨텍스트 ──▶ LLM ──▶ {답변, 인용}
```

## Stack / 스택
FastAPI · ChromaDB (vector store) · OpenAI embeddings · **any LLM** (OpenAI / Anthropic / local SLM via Ollama).

## Quickstart / 빠른 시작
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your key / 키 입력
python -m qa.indexer          # index ./docs  / 문서 색인
uvicorn qa.api:app --reload   # http://localhost:8000/docs
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"환불은 며칠 이내에 가능한가요?"}'
```

## Switch providers / 벤더 전환 (no code change)
Set in `.env`: `PROVIDER=openai` (default) · `anthropic` · `ollama` (100% local, no key).
`.env` 한 줄로 전환 — 클라우드든 로컬 SLM이든.

| Step | File | What it does / 하는 일 |
|---|---|---|
| Index | `qa/indexer.py` | load → chunk(overlap) → embed → Chroma |
| Retrieve | `qa/retriever.py` | embed query → nearest top-k chunks |
| Agent | `qa/agent.py` | context → LLM → forced-JSON answer + citations |
| API | `qa/api.py` | `POST /ask`, `GET /healthz` |
| Eval | `qa/eval.py` | golden-set citation accuracy / 인용 정확도 |

`make install · index · serve · eval · test`

## License
MIT. Educational. / 교육용.
