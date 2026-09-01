"""
src/retriever/router.py
쿼리 의도에 따라 리트리버 자동 선택

동작 흐름:
  쿼리 입력
  → LLM이 증권사명 추출 (특정 증권사 언급 시)
  → 증권사명 정규화 (대신 → 대신증권 등)
  → LLM이 의도 분류 (비교 vs 일반)
  → 특정 증권사 언급: 해당 증권사 청크만 필터링
  → 비교 쿼리: retriever_02_balanced (증권사별 균등)
  → 일반 쿼리: retriever_01_ensemble (점수 기반)
  → 선택된 리트리버로 검색 실행

분류 기준:
  balanced: 증권사별 비교/다양한 관점이 필요한 쿼리
  ensemble: 특정 사실/수치/전망을 찾는 일반 쿼리
"""

import json
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever

try:
    from . import retriever_01_ensemble as ret1
    from . import retriever_02_balanced as ret2
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.retriever import retriever_01_ensemble as ret1
    from src.retriever import retriever_02_balanced as ret2


# ── 설정 ──────────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL = "gpt-4o-mini"

_CLASSIFY_PROMPT = """다음 증권사 리포트 검색 쿼리를 분석하세요.

쿼리: "{query}"

다음 두 가지를 JSON으로 답하세요:

1. intent: 쿼리 의도
   - "balanced": 증권사별 비교, 다양한 관점이 필요한 쿼리
     예시: "증권사별 반도체 의견", "각 증권사 2차전지 전망 비교", "다양한 관점"
   - "ensemble": 특정 사실/수치/전망을 찾는 일반 쿼리
     예시: "반도체 업황 전망", "삼성전자 목표주가", "HBM 수요 현황"

2. firms: 쿼리에서 언급된 특정 증권사 이름 목록 (없으면 빈 리스트)
   - 증권사명만 추출 (예: "하나증권", "대신증권", "키움증권")
   - 언급이 없으면 []

반드시 아래 JSON 형식으로만 답하세요:
{{"intent": "balanced" 또는 "ensemble", "firms": ["증권사1", "증권사2"]}}"""


# ── 증권사 이름 정규화 ────────────────────────────────────────────────────────
# 사용자 입력 → 메타데이터 실제 이름 (source_firm)

FIRM_ALIASES = {
    # DS투자증권
    "DS투자증권": "DS투자증권",
    "DS증권": "DS투자증권",
    "ds투자증권": "DS투자증권",
    "ds증권": "DS투자증권",
    "대신증권": "DS투자증권",
    "대신": "DS투자증권",

    # IBK투자증권
    "IBK투자증권": "IBK투자증권",
    "IBK증권": "IBK투자증권",
    "ibk투자증권": "IBK투자증권",
    "ibk증권": "IBK투자증권",
    "IBK": "IBK투자증권",

    # iM증권
    "iM증권": "iM증권",
    "IM증권": "iM증권",
    "im증권": "iM증권",
    "iM": "iM증권",

    # SK증권
    "SK증권": "SK증권",
    "sk증권": "SK증권",
    "SK": "SK증권",

    # 교보증권
    "교보증권": "교보증권",
    "교보": "교보증권",


    # 유안타증권
    "유안타증권": "유안타증권",
    "유안타": "유안타증권",

    # 유진투자증권
    "유진투자증권": "유진투자증권",
    "유진증권": "유진투자증권",
    "유진": "유진투자증권",

    # 키움증권
    "키움증권": "키움증권",
    "키움": "키움증권",

    # 하나증권
    "하나증권": "하나증권",
    "하나": "하나증권",

    # 한국IR협의회
    "한국IR협의회": "한국IR협의회",
    "한국IR": "한국IR협의회",
    "IR협의회": "한국IR협의회",

    # 한화투자증권
    "한화투자증권": "한화투자증권",
    "한화증권": "한화투자증권",
    "한화": "한화투자증권",
}


def normalize_firms(firms: list[str]) -> list[str]:
    """LLM이 추출한 증권사명을 메타데이터 실제 이름으로 정규화"""
    result = []
    for firm in firms:
        normalized = FIRM_ALIASES.get(firm)
        if normalized:
            result.append(normalized)
        else:
            # 부분 매칭 시도
            matched = False
            for alias, canonical in FIRM_ALIASES.items():
                if alias in firm or firm in alias:
                    result.append(canonical)
                    matched = True
                    break
            if not matched:
                print(f"  ⚠️  알 수 없는 증권사: {firm} → 그대로 사용")
                result.append(firm)

    return list(set(result))  # 중복 제거


# ── LLM 분류기 ────────────────────────────────────────────────────────────────

_llm = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=CLASSIFIER_MODEL,
            temperature=0,
        )
    return _llm


def analyze_query(query: str) -> tuple[str, list[str]]:
    """
    쿼리 분석: 의도 분류 + 증권사명 추출 + 정규화
    Returns: (intent, firms)
      intent: "balanced" | "ensemble"
      firms:  정규화된 증권사 목록 (없으면 [])
    """
    prompt = _CLASSIFY_PROMPT.format(query=query)
    result = _get_llm().invoke(prompt).content.strip()

    try:
        parsed = json.loads(result)
        intent = parsed.get("intent", "ensemble")
        firms  = parsed.get("firms", [])

        if intent not in ("balanced", "ensemble"):
            intent = "ensemble"

        # 증권사명 정규화
        firms = normalize_firms(firms)

        return intent, firms

    except Exception:
        print(f"  ⚠️  분류 실패 → ensemble, 전체 검색")
        return "ensemble", []


def _filter_by_firms(docs: list[Document], firms: list[str]) -> list[Document]:
    """
    특정 증권사 청크만 필터링
    firms가 비어있으면 전체 반환
    증권사별로 리포트 보유 여부 확인 후 없으면 안내
    """
    if not firms:
        return docs

    # 증권사별 보유 여부 확인
    available_firms = set(
        doc.metadata.get("source_firm", "")
        for doc in docs
    )
    for firm in firms:
        if firm not in available_firms:
            print(f"  ❌ '{firm}'의 리포트가 없습니다.")
        else:
            firm_count = sum(
                1 for doc in docs
                if doc.metadata.get("source_firm") == firm
            )
            print(f"  ✅ '{firm}': {firm_count}개 청크 보유")

    filtered = [
        doc for doc in docs
        if any(
            firm in (doc.metadata.get("source_firm") or "")
            for firm in firms
        )
    ]

    if not filtered:
        print(f"  ⚠️  검색 가능한 리포트 없음 → 전체 검색으로 fallback")
        return docs

    return filtered


# ── 라우터 ────────────────────────────────────────────────────────────────────

def build_retriever(
    vectorstore,
    all_docs: list[Document],
    k:        int = 40,
) -> tuple:
    """
    두 리트리버 모두 준비
    retrieve() 호출 시 쿼리에 따라 선택

    Returns: (ret1_instance, ret2_instance, all_docs)
    """
    ret1_instance = ret1.build_retriever(vectorstore, all_docs, k=k)
    ret2_instance = ret2.build_retriever(vectorstore, all_docs, k=k)
    return ret1_instance, ret2_instance, all_docs, vectorstore


def retrieve(
    retrievers: tuple,
    query:      str,
    k:          int = 40,
) -> list[Document]:
    """
    쿼리 의도 + 증권사 언급에 따라 자동 검색 (main.py용)

    동작:
    1. LLM으로 의도 분류 + 증권사명 추출 + 정규화
    2. 의도에 따라 리트리버 선택 (앙상블 내부에서 BM25+벡터 검색)
    3. 특정 증권사 언급 시 → 검색 결과에서 해당 증권사만 필터링
    """
    docs, _, _ = retrieve_with_meta(retrievers, query, k=k)
    return docs


def retrieve_with_meta(
    retrievers: tuple,
    query:      str,
    k:          int = 40,
) -> tuple[list[Document], str, list[str]]:
    """
    쿼리 의도 + 증권사 언급에 따라 자동 검색 (freeform_chain용)

    Returns: (docs, intent, firms)
      docs:   검색된 Document 리스트
      intent: "balanced" | "ensemble"
      firms:  언급된 증권사 목록 (없으면 [])
    """
    ret1_instance, ret2_instance, all_docs, vectorstore = retrievers

    # 1. 쿼리 분석
    intent, firms = analyze_query(query)
    print(f"  쿼리 의도: {intent}")
    if firms:
        print(f"  언급된 증권사: {firms}")

    # 2. 리트리버 선택 (앙상블 내부에서 BM25+벡터 검색)
    if intent == "balanced":
        print(f"  → retriever_02_balanced 사용 (증권사별 균등 샘플링)")
        raw = ret2.retrieve(ret2_instance, query, k=k)
    else:
        print(f"  → retriever_01_ensemble 사용 (점수 기반)")
        raw = ret1.retrieve(ret1_instance, query, k=k)

    # 3. 특정 증권사 언급 시 → 검색 결과에서 필터링
    if firms:
        print(f"  → 증권사 필터링: {firms}")
        result = _filter_by_firms(raw, firms)
        print(f"  → 필터링 결과: {len(result)}개 청크")
        return result, intent, firms

    return raw, intent, firms


# ── freeform_chain 전용: 외부에서 intent 받아 리트리버만 선택 ─────────────────

def select_and_retrieve(
    retrievers: tuple,
    query:      str,
    intent:     str = "ensemble",
    k:          int = 40,
) -> list[Document]:
    """
    freeform_chain 전용 함수
    - intent는 freeform_chain._analyze_intent()가 이미 추출한 값을 받음
    - LLM 호출 없음 (리트리버 선택만)
    - 증권사 추출/필터링 없음 (freeform_chain의 target_brokers가 처리)

    Args:
        retrievers: build_retriever()가 반환한 튜플
        query:      검색 쿼리
        intent:     "balanced" | "ensemble" (freeform_chain에서 전달)
        k:          반환할 청크 수
    """
    ret1_instance, ret2_instance, all_docs, vectorstore = retrievers

    if intent == "balanced":
        print(f"  → retriever_02_balanced 사용 (증권사별 균등 샘플링)")
        return ret2.retrieve(ret2_instance, query, k=k)
    else:
        print(f"  → retriever_01_ensemble 사용 (점수 기반)")
        return ret1.retrieve(ret1_instance, query, k=k)


# ── 단독 실행 (테스트) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_queries = [
        "반도체 업황 전망",
        "증권사별 2차전지 투자 의견 비교",
        "삼성전자 목표주가",
        "각 증권사 반도체 시각이 어때",
        "HBM 수요 현황",
        "반도체에 대한 다양한 관점 알려줘",
        "하나증권이랑 대신증권의 반도체 업황 비교해줘",
        "대신증권 한화증권의 반도체 업황 비교해줘",
        "DS투자증권은 반도체를 어떻게 봐?",
        "키움 2차전지 리포트 보여줘",
        "iM증권이랑 교보 비교",
        "유안타증권 SK증권 IBK투자증권 반도체 의견",
        "한화 리포트에서 ESS 언급된 내용",
        "대신 하나 반도체 비교해줘",
    ]

    print("=" * 65)
    print("쿼리 분석 테스트 (의도 + 증권사 추출 + 정규화)")
    print("=" * 65)
    for query in test_queries:
        intent, firms = analyze_query(query)
        firms_str = ", ".join(firms) if firms else "없음"
        print(f"  [{intent:10}] 증권사: {firms_str:30} | {query}")