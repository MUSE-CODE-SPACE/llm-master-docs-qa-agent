"""The agent: retrieve evidence -> answer with citations (forced JSON).
Works with OpenAI, Anthropic, or a local SLM (Ollama) — set PROVIDER in .env.
에이전트: 근거 검색 -> 인용과 함께 답(JSON 강제). OpenAI/Anthropic/로컬 SLM(Ollama) 모두 지원."""
import json

from qa.models import AnswerResponse, Citation
from qa.retriever import retrieve
from qa.settings import settings

SYSTEM_PROMPT = (
    "You answer ONLY from the provided context. If the context does not contain "
    "the answer, say you don't know. Reply as a single JSON object with keys: "
    '"answer" (string), "citations" (list of {"source": str, "chunk_index": int}), '
    '"confidence" ("high"|"medium"|"low"). '
    "오직 주어진 컨텍스트에서만 답하라. 없으면 모른다고 답하라."
)


def build_user_prompt(question: str, chunks) -> str:
    ctx = "\n\n".join(
        f"[source #{i + 1}] (from {c.source}, chunk {c.chunk_index})\n{c.content}"
        for i, c in enumerate(chunks)
    )
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer as JSON:"


def _call_llm(system: str, user: str) -> str:
    """Return raw model text. One function, three providers. / 한 함수, 세 벤더."""
    if settings.provider == "anthropic":
        from anthropic import Anthropic

        c = Anthropic(api_key=settings.anthropic_api_key)
        r = c.messages.create(
            model=settings.llm_model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user},
                      {"role": "assistant", "content": "{"}],
        )
        return "{" + r.content[0].text
    # openai or ollama (both speak the OpenAI Chat API) / openai·ollama는 같은 API
    from openai import OpenAI

    if settings.provider == "ollama":
        c = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    else:
        c = OpenAI(api_key=settings.openai_api_key)
    r = c.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content


def answer_question(question: str) -> AnswerResponse:
    chunks = retrieve(question)
    if not chunks:
        return AnswerResponse(
            answer="I couldn't find anything relevant. / 문서에서 관련 내용을 찾지 못했어요.",
            citations=[], confidence="low", retrieved_count=0,
        )
    raw = _call_llm(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return AnswerResponse(answer=raw, citations=[], confidence="low", retrieved_count=len(chunks))
    return AnswerResponse(
        answer=data.get("answer", ""),
        citations=[Citation(**c) for c in data.get("citations", []) if isinstance(c, dict) and "source" in c],
        confidence=data.get("confidence", "medium"),
        retrieved_count=len(chunks),
    )
