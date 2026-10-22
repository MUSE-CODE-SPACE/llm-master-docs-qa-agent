"""Offline tests for the agent layer: prompt building, JSON parsing, provider calls.
오프라인 테스트: 프롬프트 생성, JSON 파싱, 벤더 호출(가짜 클라이언트)."""
import json
from types import SimpleNamespace

import anthropic
import openai

from qa import agent as agent_mod
from qa.agent import ANSWER_SCHEMA, answer_question, build_user_prompt
from qa.models import RetrievedChunk
from qa.settings import settings

CHUNKS = [
    RetrievedChunk(content="Refunds within 14 days.", source="refund_policy.md", chunk_index=0, distance=0.1)
]


def test_build_user_prompt_numbers_sources():
    p = build_user_prompt("How long?", CHUNKS)
    assert "[source #1] (from refund_policy.md, chunk 0)" in p
    assert p.rstrip().endswith("Answer as JSON:")


def test_answer_question_parses_json(monkeypatch):
    monkeypatch.setattr(agent_mod, "retrieve", lambda q: CHUNKS)
    payload = {"answer": "14 days", "citations": [{"source": "refund_policy.md", "chunk_index": 0}],
               "confidence": "high"}
    monkeypatch.setattr(agent_mod, "_call_llm", lambda s, u: json.dumps(payload))
    res = answer_question("How long?")
    assert res.answer == "14 days"
    assert res.citations[0].source == "refund_policy.md"
    assert res.confidence == "high" and res.retrieved_count == 1


def test_answer_question_handles_non_json(monkeypatch):
    monkeypatch.setattr(agent_mod, "retrieve", lambda q: CHUNKS)
    monkeypatch.setattr(agent_mod, "_call_llm", lambda s, u: "not json at all")
    res = answer_question("x")
    assert res.answer == "not json at all" and res.confidence == "low"


def test_answer_question_no_chunks(monkeypatch):
    monkeypatch.setattr(agent_mod, "retrieve", lambda q: [])
    res = answer_question("x")
    assert res.retrieved_count == 0 and res.citations == []


def test_call_llm_anthropic_uses_structured_output(monkeypatch):
    calls = []

    def create(**kw):
        calls.append(kw)
        # Thinking block first (Claude Haiku 5.5 thinks by default), text second.
        return SimpleNamespace(content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"answer": "ok", "citations": [], "confidence": "high"}'),
        ])

    class FakeAnthropic:
        def __init__(self, **kw):
            self.messages = SimpleNamespace(create=create)

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(settings, "provider", "anthropic")
    raw = agent_mod._call_llm("sys", "user")
    assert json.loads(raw)["answer"] == "ok"
    req = calls[0]
    assert req["messages"][-1]["role"] == "user"  # no assistant prefill
    assert req["output_config"]["format"]["schema"] == ANSWER_SCHEMA
    assert "temperature" not in req


def test_call_llm_openai_json_mode(monkeypatch):
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": "ok"}'))])

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(settings, "provider", "openai")
    raw = agent_mod._call_llm("sys", "user")
    assert json.loads(raw)["answer"] == "ok"
    assert calls[0]["response_format"] == {"type": "json_object"}
