"""
FABOT 전략 규칙 검색 도구 (search_strategy_rule) — RAG/임베딩 데모

CLAUDE.md의 매매 규칙을 조각(chunk)으로 나눠 Qdrant(인메모리)에 저장하고,
fastembed로 임베딩해서 질문과 의미가 비슷한 규칙을 찾아온다.
실측: 2026-08-27, qwen3.5:0.8b(로컬 Ollama), temperature=0.
실행: uv run --with qdrant-client --with fastembed --with langchain-core --with langchain-ollama python rag_strategy_search.py
"""
import asyncio
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama

# FABOT 전략 규칙 — CLAUDE.md 원문을 규칙 단위로 조각냄
STRATEGY_RULES = [
    "TQQQ 매수: F&G 지수가 25 이하이면 실탄의 25%를 매수한다. 20 이하이면 50%, 15 이하이면 100%를 매수한다. 매수 후 3거래일 쿨다운이 적용된다.",
    "TQQQ 매도: F&G 지수가 75 이상이면 보유분의 50%를 매도한다. 80 이상이면 남은 보유분 전량을 매도한다.",
    "커버드콜 추가매수: F&G 지수가 35에서 65 사이(평시 구간)이면 실탄의 10%를 커버드콜(TIGER 배당커버드콜액티브)에 투입한다. 매수 후 3거래일 쿨다운이 적용된다.",
    "비과세 배당 ETF 매집 원칙: KODEX 200, RISE 200위클리커버, TIGER 배당커버드콜액티브처럼 배당에 세금이 붙지 않는 상품은 평가손익이 마이너스여도 세후 배당수익률 관점에서 저가 매집 기회로 본다.",
    "배당소득 2000만원 제약: 1인 기준 연간 배당소득이 2000만원을 넘으면 종합과세 대상이 되고 건강보험료에도 영향을 준다. 그래서 배당을 가족 구성원 명의로 분산해서 각자 2000만원 밑으로 관리한다.",
]

COLLECTION = "fabot_strategy_rules"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_index(client: QdrantClient) -> None:
    """규칙 조각들을 임베딩해서 인메모리 Qdrant에 저장한다."""
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vectors = list(embedder.embed(STRATEGY_RULES))
    dim = len(vectors[0])

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": STRATEGY_RULES[i]})
            for i in range(len(STRATEGY_RULES))
        ],
    )


def search(client: QdrantClient, query: str, top_k: int = 2) -> list[dict]:
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=EMBED_MODEL)
    query_vector = list(embedder.embed([query]))[0].tolist()
    hits = client.query_points(collection_name=COLLECTION, query=query_vector, limit=top_k).points
    return [{"text": h.payload["text"], "score": round(h.score, 4)} for h in hits]


async def main():
    client = QdrantClient(":memory:")
    build_index(client)

    @tool
    def search_strategy_rule(query: str) -> list[dict]:
        """FABOT 매매 전략 규칙 중 질문과 의미상 가장 관련 있는 규칙을 찾아 반환합니다.
        query에는 사용자의 질문을 자연어 그대로 넣습니다.
        신규 규칙을 만들거나 규칙을 수정하지 않습니다, 조회만 합니다."""
        return search(client, query, top_k=2)

    llm = ChatOllama(model="qwen3.5:0.8b", temperature=0, reasoning=False)
    llm_with_tools = llm.bind_tools([search_strategy_rule])

    question = "커버드콜은 언제 얼마나 추가로 사?"
    r1 = await llm_with_tools.ainvoke(question)
    print("[요청]", question)
    print("[1] 모델의 요청:", [(tc["name"], tc["args"]) for tc in r1.tool_calls])

    result = search_strategy_rule.invoke(r1.tool_calls[0]["args"])
    print("[2] 하네스의 실행 결과:", json.dumps(result, ensure_ascii=False, indent=2))

    r2 = await llm_with_tools.ainvoke(
        [r1, ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=r1.tool_calls[0]["id"])]
    )
    print("[3] 결과를 읽은 최종 답:", r2.content)


if __name__ == "__main__":
    asyncio.run(main())
