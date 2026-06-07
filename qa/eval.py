"""Tiny golden-set eval: did retrieval surface the expected source?
초간단 평가: 검색이 기대한 출처를 가져왔는가?"""
from qa.agent import answer_question

GOLDEN = [
    {"q": "How many days do I have to request a refund?", "expect_source": "refund_policy.md"},
    {"q": "환불은 며칠 이내에 가능한가요?", "expect_source": "refund_policy.md"},
    {"q": "How do I reset my password?", "expect_source": "technical_faq.md"},
]


def run() -> float:
    hits = 0
    for case in GOLDEN:
        res = answer_question(case["q"])
        sources = {c.source for c in res.citations}
        ok = case["expect_source"] in sources
        hits += ok
        print(f"[{'OK ' if ok else 'MISS'}] {case['q'][:40]:40s} -> {sources or '∅'}")
    score = hits / len(GOLDEN)
    print(f"\ncitation@source = {score:.0%} ({hits}/{len(GOLDEN)})")
    return score


if __name__ == "__main__":
    run()
