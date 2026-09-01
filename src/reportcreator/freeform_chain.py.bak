"""
src/reportcreator/freeform_chain.py
자유형 질문 → 리포트 생성 체인(통합 버전)

분기 로직:
  - question_type ∈ {fact_lookup, coverage_summary, timeline, broker_comparison, risk, consensus}
      → freeform few-shot 경로
      → 공통 4섹션(핵심 요약·시장 분석·증권사별 관점 차이·주요 논거 포인트)
        + 질문 유형별 추가 섹션
  - question_type == "other" (모호하거나 복합적인 질문)
      → 풀 리포트 경로
      → 증권사별 요약 → 컨센서스/이견 → 인사이트 → 8블록 종합 리포트

지원 질문 유형:
  - fact_lookup       : "이 수치의 출처와 산출 근거가 뭐야?"
  - coverage_summary  : "특정 이벤트·종목·섹터에 대해 어떤 증권사가 언제 어떤 형식으로
                         다뤘는지 커버리지 인벤토리를 정리. 누가 심층 분석했고 누가
                         위클리성 언급에 그쳤는지 커버 깊이·방식·분석 범위 차이까지 비교."
  - timeline          : "이번 달 반도체 섹터 투자의견 변화"
  - broker_comparison : "하나증권과 키움증권의 3월 의견 차이"
  - risk              : "조선업에서 언급된 리스크 요인 정리"
  - consensus         : "AI 인프라에 대해 증권사들이 공통으로 강조하는 것"
  - other             : 위 유형에 해당하지 않는 모호한/복합 질문 → 풀 리포트 생성

사용 예시:
    from src.reportcreator.freeform_chain import answer_question

    # 명확한 질문 → freeform few-shot 경로
    result = answer_question(retriever, "하나증권과 키움증권의 3월 의견 차이를 설명해줘", ...)
    # 모호한 질문(other) → 풀 리포트 경로
    result = answer_question(retriever, "반도체 섹터 어때?", ...)
    print(result["answer"])
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from langchain.schema import Document, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.retriever.router import normalize_firms  


# 주의: retrieve/rerank 함수는 main.py 가 retrieve_fn / rerank_fn 인자로 주입한다.
# 여기서 특정 전략을 직접 import 하면 main.py 의 전략 교체(RETRIEVER / RERANKER)가 무력화되므로,
# 이 모듈은 어떤 retriever/reranker 전략과도 결합 가능하도록 함수 인자에만 의존한다.

from src.retriever.router import select_and_retrieve as _router_select_and_retrieve
from src.reranker.reranker_01_crossencoder import rerank as _default_rerank


VALID_SECTORS = {
    "건설", "건자재", "광고", "금융", "기계", "휴대폰", "담배", "유통",
    "미디어", "바이오", "반도체", "보험", "석유화학", "섬유의류", "소프트웨어",
    "운수창고", "유틸리티", "은행", "인터넷포탈", "자동차", "전기전자", "제약",
    "조선", "종이", "증권", "철강금속", "타이어", "통신", "항공운송", "홈쇼핑",
    "음식료", "여행", "게임", "IT", "에너지", "해운", "지주회사", "디스플레이",
    "화장품", "자동차부품", "교육", "기타",
}

SECTOR_ALIASES = {
    "HBM": "반도체",
    "DRAM": "반도체",
    "NAND": "반도체",
    "메모리": "반도체",
    "완성차": "자동차",
    "조선업": "조선",
    "바이오제약": "바이오",
    "2차전지": "기타",
    "배터리": "기타",
    "전기차": "기타",
    "EV": "기타",
    "양극재": "기타",
    "ESS": "기타",
}

def normalize_sector(sector: str) -> str:
    if not sector:
        return None
    if sector in SECTOR_ALIASES:
        return SECTOR_ALIASES[sector]
    if sector in VALID_SECTORS:
        return sector
    return "기타"



# ── LLM ──────────────────────────────────────────────────────────────────────

def _llm_fast()   -> ChatOpenAI: return ChatOpenAI(model="gpt-4o-mini", temperature=0)
def _llm_strong() -> ChatOpenAI: return ChatOpenAI(model="gpt-4o",      temperature=0)


# ── 메타데이터 헬퍼 (source_firm / broker 양쪽 호환) ─────────────────────────

def _get_firm(doc: Document) -> str:
    """freeform 은 source_firm, report_chain 은 broker 를 사용하므로 양쪽 호환.

    Notes:
        단순 `or` 체이닝은 빈 문자열도 falsy 로 처리하여 의도치 않은 fallback 이 발생할 수
        있으므로, 명시적 isinstance + strip 검증으로 교체한다.
        0 / False 같은 비문자열 falsy 값이 metadata 에 혼입된 경우도 방어된다.
    """
    for key in ("source_firm", "broker"):
        val = doc.metadata.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "알 수 없음"


# ── Step 1: 질문 유형 분류 + 검색 쿼리 생성 (7유형 few-shot) ─────────────────

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 증권사 리서치 리포트 검색 전략가입니다.
사용자 질문을 분석하여 JSON으로 응답하세요.

{{
  "question_type": "fact_lookup | coverage_summary | timeline | broker_comparison | risk | consensus | other",
  "target_brokers": ["언급된 증권사 목록, 없으면 빈 배열"],
  "target_sector":  "언급된 섹터/종목, 없으면 null",
  "target_period":  "언급된 기간 (예: 2026-03), 없으면 null",
  "search_queries": ["쿼리1", "쿼리2", "쿼리3"],
  "structure_hint": "질문 특성에 맞는 답변 구성 방향 한 문장"
}}

분류 기준: 
1. fact_lookup:
- asks source, basis, number origin
- keywords: 근거, 어디서, 출처, 왜 이 수치

2. coverage_summary:
- asks to organize or list reports/opinions
- keywords: 정리, 현황, 최근 리포트, 누가 뭐라고
- even if time exists → still coverage if no "change"

3. timeline:
- asks change over time (must include change)
- keywords: 변화, 흐름, 상향, 하향, 전환
- requires before → after structure

4. broker_comparison:
- requires explicit comparison
- keywords: 차이, 비교, 왜 다르게
- simple listing = NOT broker

5. risk:
- ONLY when risk alone is asked
- keywords: 리스크, 위험
- if combined with 전망/분석 → other

6. consensus:
- ONLY when explicitly asking common view
- keywords: 공통, 컨센서스, 다 같이
- otherwise DO NOT use

7. other:
- ambiguous
- mixed intent(전망 + 리스크)
- general analysis

If uncertain → return "other"

JSON만 반환(설명 없이)."""),

    # ── few-shot 1: fact_lookup ──────────────────────────────────────────
    HumanMessage(content="삼성전자 2026년 영업이익 327조원이라는 추정치 근거가 뭐야"),
    AIMessage(content=json.dumps({
        "question_type":  "fact_lookup",
        "target_brokers": [],
        "target_sector":  "삼성전자",
        "target_period":  "2026",
        "search_queries": [
            "삼성전자 2026년 영업이익 327조원 추정치 근거",
            "삼성전자 2026년 영업이익 전망 메모리 가격 장기공급계약",
            "삼성전자 실적 추정치 상향 근거 2026 영업이익",
        ],
        "structure_hint": "특정 수치의 출처 증권사와 날짜를 식별하고, 해당 수치가 어떤 가정·데이터·밸류에이션 논리로 도출됐는지 원 리포트 흐름에 맞춰 재현",
    }, ensure_ascii=False)),

    # ── few-shot 2: coverage_summary ─────────────────────────────────────
    HumanMessage(content="아블라야 FDA 허가 관련 커버리지 현황 정리해줘"),
    AIMessage(content=json.dumps({
        "question_type":  "coverage_summary",
        "target_brokers": [],
        "target_sector":  "아블라야 FDA 허가",
        "target_period":  None,
        "search_queries": [
            "아블라야 FDA 허가 증권사 리포트 커버리지",
            "아블라야 FDA 승인 리포트 현황",
            "BBB 셔틀 아블라야 FDA 가속승인 국내 증권사",
        ],
        "structure_hint": "특정 이벤트를 어떤 증권사가 언제 어떤 형식으로 다뤘는지 커버리지 인벤토리를 정리하고, 커버 깊이·방식·분석 범위 차이를 비교",
    }, ensure_ascii=False)),

    # ── few-shot 3: timeline ─────────────────────────────────────────────
    HumanMessage(content="이번 달 반도체 섹터 투자의견 변화 알려줘"),
    AIMessage(content=json.dumps({
        "question_type":  "timeline",
        "target_brokers": [],
        "target_sector":  "반도체",
        "target_period":  "2026-04",
        "search_queries": [
            "반도체 투자의견 변화 2026년 4월",
            "반도체 목표주가 상향 하향 최근",
            "반도체 업황 시각 전환 변화 흐름",
        ],
        "structure_hint": "기간 내 투자의견·목표주가·업황 해석이 어떻게 변했는지 시간 순으로 정리하고, 전환점을 유발한 트리거와 논거를 분석",
    }, ensure_ascii=False)),

    # ── few-shot 4: broker_comparison ────────────────────────────────────
    HumanMessage(content="하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘"),
    AIMessage(content=json.dumps({
        "question_type":  "broker_comparison",
        "target_brokers": ["하나증권", "키움증권"],
        "target_sector":  "반도체",
        "target_period":  "2026-03",
        "search_queries": [
            "하나증권 반도체 투자의견 3월",
            "키움증권 반도체 투자의견 3월",
            "반도체 업황 전망 하나증권 키움증권 비교",
        ],
        "structure_hint": "두 증권사의 업황 해석·핵심 변수·투자의견·목표주가 논거를 대조하고, 같은 데이터를 다르게 해석한 근본 원인을 분석",
    }, ensure_ascii=False)),

    # ── few-shot 5: risk ─────────────────────────────────────────────────
    HumanMessage(content="조선업에서 언급된 리스크 요인 정리해줘"),
    AIMessage(content=json.dumps({
        "question_type":  "risk",
        "target_brokers": [],
        "target_sector":  "조선",
        "target_period":  None,
        "search_queries": [
            "조선업 리스크 요인 불확실성",
            "조선 원가 상승 수주 지연 리스크",
            "조선 섹터 하방 시나리오",
        ],
        "structure_hint": "리스크만 단독으로 묻는 질문이므로 단기·구조적 리스크로 분류하고, 각 리스크의 발생 조건·영향 방향·증권사별 강조 차이를 정리",
    }, ensure_ascii=False)),

    # ── few-shot 6: consensus ────────────────────────────────────────────
    HumanMessage(content="AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
    AIMessage(content=json.dumps({
        "question_type":  "consensus",
        "target_brokers": [],
        "target_sector":  "AI 인프라",
        "target_period":  None,
        "search_queries": [
            "AI 인프라 데이터센터 투자 전망",
            "AI 인프라 증권사 컨센서스 수혜",
            "AI 전력 냉각 네트워크 장비 성장",
        ],
        "structure_hint": "여러 증권사가 공통으로 강조하는 논거를 수요·공급·실적·밸류체인 항목별로 정리하고, 컨센서스 배경과 아직 이견이 남은 영역을 구분",
    }, ensure_ascii=False)),

    # ── other 유형 few-shot: 모호한/종합적 질문 ──────────────────────────
    HumanMessage(content="반도체 섹터 어때?"),
    AIMessage(content=json.dumps({
        "question_type":  "other",
        "target_brokers": [],
        "target_sector":  "반도체",
        "target_period":  None,
        "search_queries": [
            "반도체 섹터 투자의견 전망",
            "반도체 업황 HBM AI 수요",
            "반도체 목표주가 실적 리스크",
        ],
        "structure_hint": "종합 리서치 리포트 — 시장현황·증권사별 분석·논거·리스크·전망·전략을 모두 포함",
    }, ensure_ascii=False)),

    HumanMessage(content="조선업 분석해줘"),
    AIMessage(content=json.dumps({
        "question_type":  "other",
        "target_brokers": [],
        "target_sector":  "조선",
        "target_period":  None,
        "search_queries": [
            "조선업 투자의견 전망",
            "조선 수주 LNG선 컨테이너선",
            "조선 목표주가 리스크 실적",
        ],
        "structure_hint": "종합 리서치 리포트 — 시장현황·증권사별 분석·논거·리스크·전망·전략을 모두 포함",
    }, ensure_ascii=False)),

    ("human", "오늘 날짜: {today}\n\n질문: {question}")
])




# ── 요청 대상 검증 헬퍼 (미지원 증권사/섹터 fallback 방지) ────────────────

def _normalize_label(value: str | None) -> str:
    """증권사/섹터/종목명 비교를 위한 간단 정규화."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value).replace("증권", "").lower()


def _target_tokens(value: str | None) -> list[str]:
    """섹터·종목명이 복합 표현일 때 검색 가능한 핵심 토큰으로 분해."""
    if not isinstance(value, str) or not value.strip():
        return []
    stopwords = {"섹터", "산업", "업종", "관련", "전망", "분석", "리포트", "의견", "관련주"}
    raw_tokens = re.split(r"[\s,/·()\[\]{}:;|+-]+", value.strip())
    tokens: list[str] = []
    for token in raw_tokens:
        token = token.strip()
        if not token or token in stopwords:
            continue
        normalized = _normalize_label(token)
        if normalized and normalized not in {_normalize_label(s) for s in stopwords}:
            tokens.append(normalized)
    normalized_full = _normalize_label(value)
    if normalized_full and normalized_full not in tokens:
        tokens.insert(0, normalized_full)
    return tokens


def _doc_search_blob(doc: Document) -> str:
    """문서가 요청 섹터/종목과 관련 있는지 확인하기 위한 비교 대상 문자열."""
    meta_keys = (
        "sector", "industry", "target_sector", "category", "sub_category",
        "ticker", "company", "title", "report_title", "filename", "source",
    )
    meta_text = " ".join(str(doc.metadata.get(k, "")) for k in meta_keys)
    # 전체 본문을 다 쓰면 비용은 없지만 비교 문자열이 커지므로 앞부분 중심으로 확인
    return _normalize_label(meta_text + " " + doc.page_content[:3000])


def _doc_matches_broker(doc: Document, broker: str) -> bool:
    target = _normalize_label(broker)
    return bool(target and target in _normalize_label(_get_firm(doc)))


def _doc_matches_sector(doc: Document, target_sector: str | None) -> bool:
    tokens = _target_tokens(target_sector)
    if not tokens:
        return True
    blob = _doc_search_blob(doc)
    # 복합 표현은 전체 표현 또는 핵심 토큰 중 하나라도 문서/메타데이터에 있으면 관련 문서로 인정
    return any(token and token in blob for token in tokens)


def _validate_and_filter_requested_scope(docs: list[Document], intent: dict) -> tuple[list[Document], str | None]:
    """
    요청한 증권사 또는 섹터가 검색 결과에 없으면 full report로 fallback하지 않고
    사용자에게 명시적인 안내 메시지를 반환하기 위한 검증 함수.
    """
    target_brokers = intent.get("target_brokers") or []
    target_sector = intent.get("target_sector")

    # 1) 증권사 검증: 요청된 증권사가 하나라도 결과에 없으면 중단
    if target_brokers:
        missing_brokers = [
            broker for broker in target_brokers
            if not any(_doc_matches_broker(doc, broker) for doc in docs)
        ]
        if missing_brokers:
            return [], "요청하신 증권사에 대한 리포트를 찾을 수 없습니다."

        # 요청 증권사가 모두 존재하면, 이후 분석에는 요청 증권사 문서만 사용
        docs = [
            doc for doc in docs
            if any(_doc_matches_broker(doc, broker) for broker in target_brokers)
        ]

    # 2) 섹터/종목 검증: 요청 섹터와 관련 있는 문서가 하나도 없으면 중단
    if target_sector:
        sector_docs = [doc for doc in docs if _doc_matches_sector(doc, target_sector)]
        if not sector_docs:
            return [], "요청하신 섹터에 관한 리포트를 찾을 수 없습니다."
        docs = sector_docs

    return docs, None




def _analyze_intent(question: str) -> dict:

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m")

    chain = _INTENT_PROMPT | _llm_fast()
    raw = chain.invoke({"question": question, "today": today}).content.strip()
    raw   = re.sub(r'^```json\s*', '', raw)
    raw   = re.sub(r'\s*```$',     '', raw)
    try:
        result = json.loads(raw)
        # ← 추가: LLM이 추출한 증권사명 정규화
        result["target_brokers"] = normalize_firms(result.get("target_brokers", []))
        return result
    except Exception:
        return {
            "question_type":  "other",
            "target_brokers": [],
            "target_sector":  None,
            "target_period":  None,
            "search_queries": [question],
            "structure_hint": "종합 리서치 리포트 — 시장현황·증권사별 분석·논거·리스크·전망·전략을 모두 포함",
        }


# ── Step 2: 다중 쿼리 검색 + 중복 제거 + Rerank ───────────────────────────────

def _collect_chunks(
    retrievers,
    queries:        list[str],
    target_brokers: list[str],
    retrieve_fn:    callable | None = None,
    rerank_fn:      callable | None = None,
    k_per_query:    int = 15,
    top_n:          int = 12,
    intent:         str = "ensemble",
    target_period:  str = None,
    target_sector:  str = None,
) -> list[Document]:
    _rerank = rerank_fn if rerank_fn else _default_rerank
    _, _, all_docs, vectorstore = retrievers

    # ── 사전 필터링 (섹터 필수 + 기간/증권사 옵션) ──────────────────────────
    if target_sector:
        pre_filtered = all_docs

        # 1. 섹터 필터링 (필수)
        sector_mapped = normalize_sector(target_sector)
        if sector_mapped:
            pre_filtered = [d for d in pre_filtered if d.metadata.get("sector", "") == sector_mapped]
            print(f"  → 사전 섹터 필터링: '{target_sector}' → '{sector_mapped}' ({len(pre_filtered)}개)")

        # 2. 기간 필터링 (옵션)
        if target_period:
            pre_filtered = [d for d in pre_filtered if d.metadata.get("report_date", "").startswith(target_period)]
            print(f"  → 사전 기간 필터링: {target_period} ({len(pre_filtered)}개)")

        # 3. 증권사 필터링 (옵션)
        if target_brokers:
            normalized = [b.replace("증권", "").replace(" ", "") for b in target_brokers]
            pre_filtered = [
                d for d in pre_filtered
                if any(nb in (_get_firm(d) or "").replace(" ", "") for nb in normalized)
            ]
            print(f"  → 사전 증권사 필터링: {target_brokers} ({len(pre_filtered)}개)")

        if not pre_filtered:
            print(f"  ⚠️ 필터링 결과 없음 → 해당 조건의 리포트가 없습니다.")
            return []  # ← 전체 검색 fallback 제거
        else:
            # 필터링된 청크로 BM25+벡터 앙상블 재구성
            try:
                from src.retriever import retriever_01_ensemble as ret1
                from src.retriever import retriever_02_balanced as ret2
            except ImportError:
                from src.retriever import retriever_01_ensemble as ret1
                from src.retriever import retriever_02_balanced as ret2

            conditions = []
            if target_sector:
                sector_mapped = normalize_sector(target_sector)
                if sector_mapped:
                    conditions.append({"sector": {"$eq": sector_mapped}})
            # target_period는 pre_filtered에서 이미 처리 → 벡터 필터 불필요
            if target_brokers:
                conditions.append({"source_firm": {"$in": target_brokers}})

            if len(conditions) == 1:
                filter_arg = conditions[0]
            elif len(conditions) > 1:
                filter_arg = {"$and": conditions}
            else:
                filter_arg = None

            ret1_new = ret1.build_retriever(vectorstore, pre_filtered, k=k_per_query, vector_filter=filter_arg)
            ret2_new = ret2.build_retriever(vectorstore, pre_filtered, k=k_per_query, vector_filter=filter_arg)

            all_candidates = []
            seen = set()
            for q in queries:
                if intent == "balanced":
                    docs_q = ret2.retrieve(ret2_new, q, k=k_per_query)
                else:
                    docs_q = ret1.retrieve(ret1_new, q, k=k_per_query)
                for doc in docs_q:
                    filename = doc.metadata.get("filename", "")
                    chunk_id = doc.metadata.get("chunk_index") or doc.metadata.get("chunk_id") or doc.page_content[:120]
                    key = (filename, chunk_id)
                    if key not in seen:
                        seen.add(key)
                        all_candidates.append(doc)

            # 증권사별 per_firm rerank (balanced + 증권사 지정)
            if target_brokers and intent == "balanced":
                per_firm = max(1, top_n // len(target_brokers))
                results = []
                for i, firm in enumerate(target_brokers):
                    nb = firm.replace("증권", "").replace(" ", "")
                    firm_docs = [
                        d for d in all_candidates
                        if nb in (_get_firm(d) or "").replace(" ", "")
                    ]
                    if firm_docs:
                        firm_query = queries[i] if i < len(queries) else queries[0]
                        reranked = _rerank(firm_query, firm_docs, top_n=per_firm)
                        results.extend(reranked)
                        print(f"  ✅ '{firm}': {len(reranked)}개 확보")
                    else:
                        print(f"  ❌ '{firm}' 청크 없음")
                return results

            combined_query = " ".join(queries)
            return _rerank(combined_query, all_candidates, top_n=top_n)

   # ── 섹터 없을 때: 기간/증권사 사전 필터링 후 검색 ──────────────────────
    if target_period or target_brokers:
        pre_filtered = all_docs

        if target_period:
            pre_filtered = [d for d in pre_filtered if d.metadata.get("report_date", "").startswith(target_period)]
            print(f"  → 사전 기간 필터링: {target_period} ({len(pre_filtered)}개)")

        if target_brokers:
            normalized = [b.replace("증권", "").replace(" ", "") for b in target_brokers]
            pre_filtered = [
                d for d in pre_filtered
                if any(nb in (_get_firm(d) or "").replace(" ", "") for nb in normalized)
            ]
            print(f"  → 사전 증권사 필터링: {target_brokers} ({len(pre_filtered)}개)")

        if not pre_filtered:
            print(f"  ⚠️ 필터링 결과 없음 → 전체 청크로 검색")
            pre_filtered = all_docs

        try:
            from src.retriever import retriever_01_ensemble as ret1
            from src.retriever import retriever_02_balanced as ret2
        except ImportError:
            from src.retriever import retriever_01_ensemble as ret1
            from src.retriever import retriever_02_balanced as ret2

        conditions = []
        if target_period:
            pass  # pre_filtered에서 처리됨
        if target_brokers:
            conditions.append({"source_firm": {"$in": target_brokers}})

        filter_arg = conditions[0] if len(conditions) == 1 else {"$and": conditions} if conditions else None

        ret1_new = ret1.build_retriever(vectorstore, pre_filtered, k=k_per_query, vector_filter=filter_arg)
        ret2_new = ret2.build_retriever(vectorstore, pre_filtered, k=k_per_query, vector_filter=filter_arg)

        all_candidates = []
        seen = set()
        for q in queries:
            if intent == "balanced":
                docs_q = ret2.retrieve(ret2_new, q, k=k_per_query)
            else:
                docs_q = ret1.retrieve(ret1_new, q, k=k_per_query)
            for doc in docs_q:
                filename = doc.metadata.get("filename", "")
                chunk_id = doc.metadata.get("chunk_index") or doc.metadata.get("chunk_id") or doc.page_content[:120]
                key = (filename, chunk_id)
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(doc)

        if target_brokers and intent == "balanced":
            per_firm = max(1, top_n // len(target_brokers))
            results = []
            for i, firm in enumerate(target_brokers):
                nb = firm.replace("증권", "").replace(" ", "")
                firm_docs = [d for d in all_candidates if nb in (_get_firm(d) or "").replace(" ", "")]
                if firm_docs:
                    firm_query = queries[i] if i < len(queries) else queries[0]
                    reranked = _rerank(firm_query, firm_docs, top_n=per_firm)
                    results.extend(reranked)
                    print(f"  ✅ '{firm}': {len(reranked)}개 확보")
                else:
                    print(f"  ❌ '{firm}' 청크 없음")
            return results

        combined_query = " ".join(queries)
        return _rerank(combined_query, all_candidates, top_n=top_n)

    # ── 섹터/기간/증권사 모두 없을 때: 전체 검색 ────────────────────────────
    all_candidates: list[Document] = []
    seen: set[tuple] = set()

    for q in queries:
        _retrieve = retrieve_fn if retrieve_fn else lambda r, _q, k, _i=intent: _router_select_and_retrieve(r, _q, intent=_i, k=k)
        for doc in _retrieve(retrievers, q, k=k_per_query):
            filename = doc.metadata.get("filename", "")
            chunk_id = (
                doc.metadata.get("chunk_index")
                or doc.metadata.get("chunk_id")
                or doc.page_content[:120]
            )
            key = (filename, chunk_id)
            if key not in seen:
                seen.add(key)
                all_candidates.append(doc)

    combined_query = " ".join(queries)
    return _rerank(combined_query, all_candidates, top_n=top_n)
# ── Step 3: 증권사별 컨텍스트 구성 (few-shot 경로용) ─────────────────────────

def _build_context(docs: list[Document], target_brokers: list[str]) -> str:
    broker_chunks: dict[str, list[str]] = {}
    broker_dates:  dict[str, set[str]]  = {}

    for doc in docs:
        firm = _get_firm(doc)
        date = doc.metadata.get("report_date", "")
        broker_chunks.setdefault(firm, []).append(doc.page_content)
        broker_dates.setdefault(firm, set())
        if date:
            broker_dates[firm].add(date)

    if target_brokers:
        normalized = [b.replace("증권", "").replace(" ", "") for b in target_brokers]
        order = sorted(
            broker_chunks.keys(),
            key=lambda f: 0 if any(nb in f.replace(" ", "") for nb in normalized) else 1,
        )
    else:
        order = list(broker_chunks.keys())

    parts = []
    for firm in order:
        date_str = ", ".join(sorted(broker_dates[firm])) or "날짜 미상"
        # ✅ 청크 수에 따라 예산을 균등 배분 — 단순 joined[:4000] 은 앞쪽 청크만 살아남는 문제 방지
        # hallucination 방지를 위해 청크별 최소 문맥을 400자로 확대하고 전체 예산도 8,000자로 확대한다.
        chunks      = broker_chunks[firm]
        budget      = 8000
        per_chunk   = max(budget // len(chunks), 400)  # 최소 400자 보장
        content     = "\n\n---\n\n".join(c[:per_chunk] for c in chunks)
        parts.append(f"### [{firm}] (리포트 날짜: {date_str})\n{content}")

    return "\n\n".join(parts)


# ── Step 4: 답변 생성 — 명확 유형 freeform few-shot ───────────────────────────
# 모든 freeform 답변에 공통 4개 섹션 포함:
#   ## 핵심 요약
#   ## 시장 분석
#   ## 증권사별 관점 차이
#   ## 주요 논거 포인트

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 15년 경력의 금융 리서치 전문 애널리스트입니다.
제공된 증권사 리포트 발췌문을 바탕으로 질문에 답변하는 리포트를 작성하세요.

**모든 답변에 반드시 포함할 4개 섹션 (순서 고정)**

1. ## 핵심 요약
   결론을 3~5문장으로 압축. 상세 논거는 아래 섹션에서 다룸.
   증권사 간 결론이 다르면 "A는 ~, B는 ~" 형식으로 대조.
   가장 중요한 수치 1~2개를 본문에 인용.

2. ## 시장 분석
   현재 시장 상황과 구조적 배경을 수치와 함께 서술.
   개별 증권사의 투자의견·결론은 다음 섹션에서 다루고,
   여기서는 시장 사실(수치, 사건, 추세)을 종합 서술한다.
   인용하는 수치·사실에는 반드시 출처 증권사명과 날짜를 표기할 것(예: "(하나증권, 2026-04-27)").

3. ## 증권사별 관점 차이
   같은 현상을 어떻게 다르게 해석하는지 시각 차이에 집중.
   핵심 변수 선택·인과관계 해석·리스크 가중치 차이를 대조.
   수치 나열보다 논리 흐름 차이를 서술.

4. ## 주요 논거 포인트
   각 주장을 뒷받침하는 구체적 수치·데이터·가정에 집중.
   해석 방향 반복 금지. 발췌문에 있는 수치만 인용.
   논거 항목 단위로 번호(①, ②, ③)를 매기고, 각 항목 아래에 증권사별 수치를 정리.
   공통 논거와 증권사별 고유 논거를 구분할 것.

**추가 섹션**은 질문 유형에 맞게 자유롭게 설계하되, [구조 가이드]에서 제시된 방향을 우선 따르세요.
필요한 경우 시계열 정리(timeline), 시나리오 분류(risk), 증권사 비교(broker_comparison) 등을 활용할 수 있습니다.
추가 섹션을 필수로 출력해야 하는 것이 아니므로, 질문 유형에 따라 필요한 경우에만 출력하세요.

**절대 원칙**
- 발췌문의 구체적 수치(목표주가, 성장률, EPS 등) 반드시 인용
- 모든 수치·사실 인용에는 출처 증권사명과 날짜를 표기(시장 분석 섹션 포함, 전 섹션 공통)
- 일반론·상식 금지, 이 발췌문에 있는 내용에만 집중
- 발췌문에 없는 정보는 "리포트에서 확인되지 않음"으로 명시
- 증권사 간 관점 차이가 없는 경우 "관점 차이 없음"으로 명시하고 공통적으로 강조하는 논거를 대신 서술할 것
- 발췌문이 단일 증권사에서만 수집된 경우 "타 증권사 의견은 수집된 리포트에서 확인되지 않음"으로 명시하고 해당 증권사 분석만 제공할 것
- few-shot 예시는 답변의 깊이와 형식 참고용이며, 실제 발췌문에 없는 인과관계·시장 함의·시나리오는 모방하지 말 것
- 해석이 필요한 경우 반드시 발췌문에 있는 수치·문장에 근거한 "발췌문 기반 해석"으로만 제한할 것
- 마크다운으로 작성"""),

# ── few-shot 1: fact_lookup ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 특정 수치의 출처를 식별하고, 해당 수치가 어떤 가정과 논거를 기반으로 도출되었는지 원 리포트의 논리 흐름을 그대로 재현
[질문] 삼성전자 2026년 영업이익 327조원이라는 추정치 근거가 뭐야
[발췌]
### [SK증권] (리포트 날짜: 2026-04-08)
삼성전자의 2026년 영업이익을 327조원으로 전망한다.
목표주가 상향 근거는 메모리 가격 상승에 따른 실적 전망 상향, 장기공급계약 가시화를 통한 실적 안정성 제고이다.
Target P/E는 기존 13.0배에서 11.0배로 하향 적용.

### [대신증권] (리포트 날짜: 2026-04-06)
메모리반도체 가격 전망 상향을 반영하여 삼성전자 2026년 영업이익 전망을 298.5조원으로 상향.
AI 수요 확대와 제한적 공급 증가로 극한의 수급 불균형이 발생하며 메모리 이익이 폭증하는 구조."""),

("ai", """\
## 핵심 요약

삼성전자 2026년 영업이익 327조원 추정치는 SK증권(2026-04-08)에서 제시된 수치로, 메모리 가격 상승과 장기공급계약 확산에 따른 실적 상향을 반영한 결과다.  
같은 시점 대신증권(2026-04-06) 역시 메모리 가격 전망 상향을 근거로 영업이익을 298.5조원으로 상향 제시하며 유사한 방향성을 보인다.  
두 증권사 모두 AI 수요 기반 메모리 업황 개선을 공통 전제로 하고 있으며, SK증권은 여기에 장기공급계약에 따른 실적 안정성을 추가 반영해 더 높은 수치를 도출했다.

## 시장 분석

2026년 반도체 시장은 AI 수요 확대와 공급 제약이 결합된 구조적 수급 불균형 국면에 있다.  
AI 추론 확장과 제한적 공급 증가로 메모리 가격 상승 압력이 강화되며 이익이 급증하는 구조가 형성되고 있다(대신증권, 2026-04-06).  
또한 장기공급계약 확산을 통해 실적 변동성이 완화되고 중기적인 안정 성장 경로가 확보되는 환경이 조성되고 있다(SK증권, 2026-04-08).

## 증권사별 관점 차이

두 증권사는 동일한 업황 개선을 전제로 하지만, **이익 추정치 반영 강도와 안정성 가정에서 차이**가 발생한다.

| 구분 | SK증권 (2026-04-08) | 대신증권 (2026-04-06) |
|------|---------------------|---------------------|
| 삼성전자 2026E 영업이익 | **327조원** | **298.5조원** |
| 핵심 변수 | 메모리 가격 상승 + 장기공급계약 | 메모리 가격 상승 |
| 안정성 반영 | 장기계약 기반 실적 안정성 반영 | 수급 기반 이익 확대 중심 |

**SK증권 — 안정성까지 반영한 상향 추정**
메모리 가격 상승에 따른 실적 상향에 더해, 장기공급계약 가시화를 통해 실적 변동성이 낮아지는 구조를 반영한다.  
이를 기반으로 Target P/E를 13.0배에서 11.0배로 하향 적용하면서도, 이익 추정치를 크게 상향하여 327조원을 도출했다.

**대신증권 — 수급 기반 이익 확대 중심**
AI 수요 확대와 제한적 공급 증가로 발생한 극한의 수급 불균형을 핵심 변수로 설정한다.  
메모리 가격 상승이 이익 폭증으로 직접 연결된다는 구조를 반영하여 298.5조원의 영업이익을 제시했다.

## 주요 논거 포인트

① 영업이익 추정치 (직접 출처)  
- SK증권 (2026-04-08): 삼성전자 2026년 영업이익 **327조원**  
- 대신증권 (2026-04-06): 삼성전자 2026년 영업이익 **298.5조원**

② 메모리 가격 상승 → 실적 상향 (공통 근거)  
- SK증권: 메모리 가격 상승 반영 → 실적 전망 상향 (SK증권, 2026-04-08)  
- 대신증권: 메모리 가격 전망 상향 반영 → 영업이익 상향 (대신증권, 2026-04-06)

③ 수급 구조 (대신증권 중심 논거)  
- AI 수요 확대 + 제한적 공급 증가 → 극한의 수급 불균형 형성 (대신증권, 2026-04-06)  
- 해당 구조가 메모리 이익 폭증으로 연결

④ 장기공급계약 효과 (SK증권 추가 논거)  
- 장기공급계약 확산 → 실적 변동성 완화 및 안정 성장 경로 확보 (SK증권, 2026-04-08)  
- 업사이클/다운사이클 모두에서 가격 방어력 확보

⑤ 밸류에이션 반영 방식 (SK증권)  
- Target P/E: **13.0배 → 11.0배 하향 적용** (SK증권, 2026-04-08)  
- 보수적 멀티플 적용에도 불구하고 이익 상향 폭이 더 커 목표주가 상향으로 연결

##추가 섹션

"""),

    # ── few-shot 2: coverage_summary ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 특정 이벤트에 대한 증권사 커버리지 현황을 정리. 누가 언제 어떤 형태로 다뤘는지 정리하고, 커버 방식과 깊이 차이를 비교
[질문] 아블라야 FDA 허가 관련 커버리지 현황 정리해줘
[발췌]
### [iM증권] (리포트 날짜: 2026-03-27)
드날리 테라퓨틱스의 헌터증후군 치료제 아블라야 FDA 가속승인.
BBB 통과 최초 상업화 사례. CSF HS 91% 감소, 환자 93% 정상 범위 도달.
국내 BBB 셔틀 관련 기업 정리 포함.

### [키움증권] (리포트 날짜: 2026-03-30)
Weekly 제약/바이오 리포트.
아블라야 FDA 승인 해외 뉴스 언급 포함.
산업 일정 중심 정리."""),

("ai", """\
## 핵심 요약

아블라야 FDA 허가 이벤트는 2026년 3월 말 기준 **총 2개 증권사에서 커버**되고 있으며, 커버 방식은 명확히 이원화된다.  
iM증권(2026-03-27)은 단독 이벤트 코멘트 리포트를 통해 임상 데이터와 산업 함의를 포함한 **심층 커버**를 제공한 반면,  
키움증권(2026-03-30)은 Weekly 리포트 내 해외 뉴스 항목으로 **간접 커버**에 그쳤다.  
현재 커버리지는 **“심층 1건 + 위클리 1건”의 제한적 인벤토리 구조**로 확인된다.

## 시장 분석

아블라야는 드날리 테라퓨틱스의 헌터증후군 치료제로 FDA 가속승인을 받았으며,  
BBB(뇌혈관장벽)를 통과하는 바이오의약품의 첫 상업화 사례로 언급된다(iM증권, 2026-03-27 / 키움증권, 2026-03-30).  
임상 결과는 두 증권사에서 공통적으로 인용되며, CSF 헤파란 황산염 91% 감소, 환자 93% 정상 범위 도달이 핵심 데이터다(iM증권, 2026-03-27 / 키움증권, 2026-03-30).  
해당 이벤트는 CNS 치료 영역에서 BBB 셔틀 플랫폼 상업화 가능성을 확인한 사례로 정리된다.

## 증권사별 관점 차이

본 케이스는 해석 차이보다는 **커버 구조 및 리포트 성격 차이**가 핵심이다.

| 증권사 | 발간일 | 리포트 유형 | 커버 깊이 | 커버 범위 |
|--------|--------|-------------|------------|------------|
| iM증권 | 2026-03-27 | 이벤트 코멘트 | **심층 분석** | 임상 데이터 + 산업 함의 + 국내 기업 |
| 키움증권 | 2026-03-30 | Weekly | **간접 커버** | 해외 뉴스 + 산업 일정 |

**iM증권 (2026-03-27) — 이벤트 중심 심층 커버**
아블라야를 독립 리포트로 다루며, 임상 데이터(CSF HS 91%, 환자 93%)와 BBB 플랫폼 구조를 상세 설명한다.  
또한 국내 BBB 셔틀 관련 기업을 별도로 정리하며 이벤트를 테마 확장 관점에서 해석한다.

**키움증권 (2026-03-30) — 산업 흐름 내 간접 커버**
Weekly 리포트 내 해외 뉴스로 포함되어 있으며, 이벤트 자체 분석보다는 산업 일정 및 동향 흐름 속에서 위치만 제시한다.  
수집된 발췌문에서는 개별 이벤트 해석이나 산업 함의 분석이 확인되지 않는다.

## 주요 논거 포인트

① 커버리지 인벤토리  
- 총 2개 증권사 커버 (iM증권, 2026-03-27 / 키움증권, 2026-03-30)

② 커버 형태 분포  
- 단독 이벤트 심층 커버: 1건 (iM증권, 2026-03-27)  
- 위클리/간접 커버: 1건 (키움증권, 2026-03-30)

③ 공통 인용 데이터  
- CSF HS 91% 감소, 환자 93% 정상 범위 도달 (iM증권, 2026-03-27 / 키움증권, 2026-03-30)

④ 리포트 구성 차이  
- iM증권: 임상 데이터 + BBB 플랫폼 + 국내 기업 리스트 포함 (iM증권, 2026-03-27)  
- 키움증권: 해외 뉴스 및 산업 일정 중심 구성 (키움증권, 2026-03-30)

⑤ 커버리지 구조의 의미  
동일 이벤트에 대해 iM증권은 **테마 확장형 심층 분석**, 키움증권은 **정보 업데이트형 위클리 커버**를 선택한 구조다.  
이는 해당 이벤트가 아직 시장 전반에서 광범위하게 커버되기보다는 일부 증권사 중심으로 제한적으로 분석되고 있음을 시사한다.

##추가 섹션
"""),

    # ── few-shot 3: timeline ──────────────────────────────────────────────
    ("human", """\
[구조 가이드] 동일 증권사의 시각이 기간 내 어떻게 변화했는지 시간 순으로 정리하고, 전환점을 유발한 트리거와 논거를 분석
[질문] 유진투자증권의 정유/석유화학 섹터 시각이 3월 말 이후 어떻게 바뀌었는지 정리해줘
[발췌]
### [유진투자증권] (리포트 날짜: 2026-03-31, 제목: "여전히 보수적")
WTI는 다시 100달러/배럴 상회. 호르무즈 해협 봉쇄로 중동 원유 수출 차질 발생.
세계 원유 재고 29억 배럴까지 감소. 세계 정유시설 가동 중단 약 900만 b/d (전월 대비 +400만 b/d 증가).
마진 확대 중이나 국내는 가격 상한제 시행 + 수출 물량 감소로 2Q부터 실적 개선 효과 미미할 것.
단기간 중동 물류 차질 해소되지 않으면 정유/석유화학 공장 가동률 조정 가능성.
"아직까지는 국내 정유/석유화학 업종에 대해 보수적 시각 유지할 필요."

### [유진투자증권] (리포트 날짜: 2026-04-07, 제목: "수요 감소 외에는 답이 안보이는 상황")
WTI +11.2% 상승하여 $103.1 도달.
호르무즈 해협 봉쇄는 세계 원유 공급의 15%를 차지 — 역사적으로 가장 큰 공급 차질 규모.
역사적 전쟁 사례 분석: 1차 오일쇼크 시 생산 -7%·유가 4배, 러우 전쟁 시 -0.1%·유가 32% 후 안정.
미국-이란 협상 뉴스 나오나 실제 선박 통항으로 이어지지 않음.
전략비축유 방출과 OPEC 증산은 임시 방편에 불과.
"국제유가 안정을 위해서는 수요 침체로 인한 수급 안정화 밖에 답이 없어 보인다."

### [유진투자증권] (리포트 날짜: 2026-04-14, 제목: "불확실성이 점점 해소되는 중")
미국-이란 협상이 물밑에서 진행 중이라는 보도로 국제유가 다시 안정화 조짐.
트럼프의 조기 타결 유인: 지지율 하락, 중간선거 부담, 소비자물가 상승.
4/6 이스라엘의 이란 석유화학단지 공격, 4/7 사우디 SABIC 단지 공격.
중동 정유/석유화학 처리 손실 비율 63%. 유전·화학 설비 복구에 최소 3~6개월 소요.
화학 제품: 2Q 재고 소진 강세 → 3Q부터 공급 점진적 회복 전망.
PE·PP·MEG 등 만성 공급과잉 제품도 가격 강세 가능성.
"현 시점에서는 비관론보다 낙관론으로 정유/화학 업종을 바라볼 필요가 있다고 판단."""),

("ai", """\
## 핵심 요약

유진투자증권은 3주에 걸쳐 정유/석유화학 섹터 시각을 **보수적 → 비관적 심화 → 낙관 전환**으로 변경했다.
3월 말(2026-03-31)에는 국내 가격 상한제로 인한 실적 개선 제약을 근거로 보수적 시각을 유지했고,
4월 초(2026-04-07)에는 호르무즈 해협 봉쇄(세계 공급의 **15%**)가 역사적 최대 공급 차질이라며 "**수요 침체 외에 답이 없다**"는 비관론을 제시했다.
그러나 4월 중순(2026-04-14)에는 미국-이란 협상 진전과 중동 처리 손실 **63%**로 인한 재고 소진 사이클을 근거로 "**비관론보다 낙관론**"으로 명시적 전환했다.

## 시장 분석

2026년 3월 말부터 4월 중순까지 정유/석유화학 시장은 **공급 차질 심화 → 협상 가시화**의 두 국면을 거쳤다.
유가 측면에서는 WTI가 3/27 $93.0(유진투자증권, 2026-03-31)에서 4/2 $103.1(유진투자증권, 2026-04-07)로 +11.2% 급등 후,
4/13에는 안정화 조짐을 보였다(유진투자증권, 2026-04-14).
공급 차질 규모는 3월 말 정유시설 가동 중단 900만 b/d(유진투자증권, 2026-03-31)에서 시작해,
4/6 이란 석유화학단지·4/7 사우디 SABIC 단지 공격으로 중동 처리 손실 비율 63%까지 확대됐다(유진투자증권, 2026-04-14).
이 같은 공급 차질은 역사적으로 가장 큰 규모(세계 공급의 15%)로 평가된다(유진투자증권, 2026-04-07).

## 증권사별 관점 차이

본 분석은 단일 증권사(유진투자증권)의 시간에 따른 시각 변화를 추적한 것으로, 타 증권사 의견은 수집된 리포트에서 확인되지 않음.
유진투자증권 내부의 시계열 시각 전환은 다음과 같다.

| 날짜 | 제목 | 시각 | 핵심 트리거 |
|------|------|------|-------------|
| 2026-03-31 | 여전히 보수적 | 보수적 유지 | 국내 가격 상한제 → 실적 개선 제약 |
| 2026-04-07 | 수요 감소 외에는 답이 안보이는 상황 | **비관론 심화** | 호르무즈 봉쇄(세계 공급 15%), 임시 방편 한계 |
| 2026-04-14 | 불확실성이 점점 해소되는 중 | **낙관론 전환** | 미·이란 협상 진전 + 재고 소진 사이클 |

**3월 말 (2026-03-31) — 보수적 유지 단계**
유가 급등과 마진 확대를 인정하면서도, 국내 가격 상한제와 수출 물량 감소로 2Q 실적 개선 효과가 제한적이라고 판단했다.
공급 차질의 장기화 가능성을 우려 요인으로 지목했다.

**4월 초 (2026-04-07) — 비관론 심화 단계**
공급 차질의 역사적 규모(세계 공급의 15%)를 강조하며, 과거 전쟁 사례 비교를 통해 이번 사태가 1차 오일쇼크 수준의 충격임을 시사했다.
협상·증산 등 어떤 정책 대응도 임시 방편에 불과하다며, "수요 침체밖에 답이 없다"는 표현으로 비관 강도를 최고조로 끌어올렸다.

**4월 중순 (2026-04-14) — 낙관 전환 단계**
시각의 명시적 전환점이다. 같은 공급 차질 데이터(처리 손실 63%)를 보면서도 해석을 정반대로 뒤집었다 — 차질이 클수록 재고 소진 사이클이 강해져 화학 제품 가격 강세가 지속된다는 논리.
"비관론보다 낙관론으로 바라볼 필요"라는 표현으로 시각 전환을 명시했다.

## 주요 논거 포인트

**① 유가·공급 차질 데이터 — 3주 연속 악화 (시계열 공통 흐름)**
- 2026-03-31: WTI $93.0 (3/27 종가), 정유시설 가동 중단 900만 b/d
- 2026-04-07: WTI $103.1 (4/2 종가, +11.2%), 호르무즈 봉쇄가 세계 공급의 15% 차지
- 2026-04-14: 중동 정유/석유화학 처리 손실 63%, 설비 복구 3~6개월 소요

**② 보수론의 근거 (유진투자증권, 2026-03-31)**
- 국내 가격 상한제 시행 → 마진 확대 효과가 실적으로 연결되지 못함
- 수출 물량 감소로 2Q 실적 개선 효과 미미할 전망
- 단기간 물류 차질 해소되지 않으면 가동률 조정 가능성

**③ 비관 심화의 근거 (유진투자증권, 2026-04-07)**
- 호르무즈 봉쇄가 세계 원유 공급의 15% — **역사적 최대 규모**
- 1차 오일쇼크(생산 -7%, 유가 4배) 대비 현재 사태의 잠재력 시사
- 전략비축유·OPEC 증산은 임시 방편 → 수요 침체만이 유일한 수급 균형 경로

**④ 낙관 전환의 근거 (유진투자증권, 2026-04-14)**
- 미국-이란 협상 물밑 진행 + 트럼프의 조기 타결 정치적 유인 (지지율·중간선거·물가)
- 처리 손실 63% → 재고 소진 강세 → **2Q 화학 제품 가격 강세, 3Q 점진적 회복**
- PE·PP·MEG 등 만성 공급과잉 제품도 가격 강세 가능성
- "전쟁 후 강한 재고 비축이 항상 이어졌다"는 역사적 패턴 언급 (발췌문에서 직접 확인되지 않는 일반화 — 유진투자증권의 서술을 확장 해석한 것임)

**⑤ 시각 전환의 근본 메커니즘**
3주 사이 공급 차질 데이터는 **계속 악화**(900만 b/d → 처리 손실 63%)했지만 시각은 **반전**됐다.
전환점은 데이터 변화가 아니라 **해석 프레임의 전환**이다 — 4/7까지는 "공급 차질 = 가동률 압박·실적 제약" 프레임, 4/14는 "공급 차질 = 재고 소진·가격 강세" 프레임으로 바뀌었다.
협상 가시화로 "공급 차질이 일시적"이라는 전제가 추가되면서, 같은 데이터가 정반대 함의를 갖게 된 구조다.

## 추가 섹션
"""),

    # ── few-shot 4: broker_comparison ────────────────────────────────────
    ("human", """\
[구조 가이드] 두 증권사의 업황 해석과 핵심 논거를 대조하고, 관점 차이의 근본 원인을 분석
[질문] 2026년 4월 반도체 업황에 대한 하나증권과 키움증권의 관점 차이를 설명해줘
[발췌]
### [하나증권] (리포트 날짜: 2026-04-27)
반도체 및 소부장 비중확대 지속 추천. 메모리 반도체 수출은 전년 대비 223% 증가한 8.4억 달러를 기록했다.
DRAM은 전년 대비 345%, NAND는 225% 증가하며 가격 상승과 수요 회복이 동시에 나타나고 있다.
모바일 DRAM 및 eSSD 가격이 예상보다 양호하며 2분기·연간 실적 상향이 지속될 것으로 판단한다.
이번 싸이클의 핵심 변수는 AI 서버/데이터센터향 LPDDR이며, 2027년 신규 Fab 기대로 소부장 동반 추천.

### [키움증권] (리포트 날짜: 2026-04-27)
반도체 Overweight (Maintain). 5월 상반월까지 업황 온기 속 종목별 차별화 흐름 전망.
대만 메모리 업종 매출은 전월 대비 21% 증가, 전년 대비 195% 증가했다.
반도체 유통 매출은 전월 대비 53% 증가했고, MediaTek 매출은 전월 대비 62% 증가하며
일부 모바일 고객의 pull-in 수요와 계절적 성수기 영향이 반영된 것으로 판단한다.
4월은 엔비디아 루빈 출하 본격화 시기로 종목별 차별화 흐름이 이어질 것."""),

("ai", """\
## 핵심 요약

두 증권사 모두 4월 27일 기준 반도체 섹터에 대해 **비중확대(하나증권) / Overweight(키움증권)** 의견을 유지한다.
다만 분석 프레임이 다르다. 하나증권은 국내 수출 데이터(+223% YoY)를 근거로 **LPDDR 단일 변수에 의한 실적 사이클 동행**을 주장하고,
키움증권은 대만 Tech 전수 매출 데이터(+195% YoY)를 근거로 **종목별 차별화 흐름**을 전망한다.

## 시장 분석

2026년 4월 반도체 시장은 메모리 중심의 수요 회복과 가격 상승이 동시에 나타나며 실적 개선 구간에 진입한 상황이다.
한국 메모리 수출은 전년 대비 +223% (하나증권, 2026-04-27), 대만 메모리 매출은 전년 대비 +195% (키움증권, 2026-04-27)로
양 지역 모두 세 자릿수 증가율이 확인된다.
반도체 유통 매출이 전월 대비 +53% (키움증권, 2026-04-27)로 가장 큰 폭의 단기 모멘텀을 보이며,
공급망 전반의 실적 개선이 데이터로 가시화되는 국면이다.

## 증권사별 관점 차이

| 구분 | 하나증권 | 키움증권 |
|------|-----------|-----------|
| 섹터 의견 | 비중확대 지속 | Overweight (Maintain) |
| 분석 프레임 | 국내 수출·실적 추정 | 대만 Tech 전수 매출 데이터 |
| 핵심 변수 | LPDDR (단일 지목) | 종목별 차별화 (다층 구조) |
| 주가 시계 | 실적 상향 동행 | 5월까지 온기 후 종목별 분기 |

**하나증권 (2026-04-27) — 단일 변수 + 실적 동행 프레임**
이번 싸이클의 변수를 AI 서버/데이터센터향 LPDDR 하나로 압축한다.
모바일 DRAM·eSSD의 단기 양호 시그널은 인정하되, 사이클을 끌고 갈 진짜 변수는 LPDDR이라는 시각이다.
실적 상향 마무리 전까지 주가도 동행한다는 단순 인과를 강조한다.

**키움증권 (2026-04-27) — 전수 데이터 + 종목별 분기 프레임**
대만 메모리·유통·LSI 등 밸류체인 전 섹터의 월별 매출액을 횡단 비교한다.
이 접근에서는 같은 강세 안에서도 매출 증가 폭이 종목별로 크게 갈리는 점이 자연스럽게 부각되며,
주가 흐름도 종목별 차별화로 귀결된다는 결론에 이른다.

## 주요 논거 포인트

**① 메모리 강세 — 데이터로 확인 (양사 공통)**
- 하나증권 (2026-04-27): 메모리 수출 +223% YoY, DRAM +345%, NAND +225%
- 키움증권 (2026-04-27): 대만 메모리 매출 +21% MoM, +195% YoY, 반도체 유통 +53% MoM

**② LPDDR 단일 변수론 (하나증권, 2026-04-27)**
- 모바일 DRAM·eSSD 가격 양호 → 단기 양호 시그널
- 핵심 변수는 AI 서버/데이터센터향 **LPDDR** — 사이클 방향을 결정
- 2027년 신규 Fab 기대 → 소부장 동반 비중확대

**③ Pull-in 수요 + 계절성 (키움증권, 2026-04-27)**
- MediaTek 매출 +62% MoM — 모바일 고객의 pull-in 수요 추정
- 4월 엔비디아 루빈 출하 본격화 + TSMC 계절적 성수기
- 5월 상반월(4월 실적 발표)까지 업황 온기 후 종목별 차별화

**④ 관점 차이의 근본 원인**
같은 +200% 수준의 YoY 데이터를 보면서도, 하나증권(2026-04-27)은 **실적 상향 → 주가 동행**의 단순 인과로,
키움증권(2026-04-27)은 **데이터 격차 → 주가 차별화**로 해석한다.
이 차이는 **데이터 소스(국내 수출 vs 대만 Tech 전수)와 분석 단위(섹터 평균 vs 종목별 매출)**에서 비롯된다.

## 추가 섹션"""),

    # ── few-shot 5: risk ──────────────────────────────────────────────────
    ("human", """\
[구조 가이드] 리스크를 단기·구조적으로 분류하고, 각 리스크가 어떤 상황에서 언급되었는지와 영향 방향을 정리한다.  
가능한 경우 증권사별로 강조하는 리스크의 차이를 비교하되, 발췌문에 근거가 없는 내용은 확장하지 않는다.
[질문] 메모리 반도체 업황에서 언급된 리스크 요인 정리해줘
[발췌]
### [유진투자증권] (리포트 날짜: 2026-04-06)
중국 선전 화창베이 시장에서 DDR5 현물가 약 30% 급락 보도.
실제 신품 가격은 안정적이나 일부 딜러 차익 실현으로 단기 약세 발생.
최근 한 달간 메모리 섹터 투자심리 일부 약화.
호르무즈 불안 등 매크로 불확실성이 주가 약세 주요 원인.

### [대신증권] (리포트 날짜: 2026-04-06)
AI 수요 확대와 제한적 공급 증가로 메모리 수급 불균형 심화.
메모리 가격 상승 기반 이익 확대 구조 형성.

### [SK증권] (리포트 날짜: 2026-04-08)
장기공급계약 가시화로 실적 안정성 제고.
다만 밸류에이션 측면에서 Target P/E를 13.0배 → 11.0배로 하향 적용."""),

("ai", """\
## 핵심 요약

메모리 반도체 업황의 리스크는 단기적으로는 DDR5 현물가격 변동성과 투자심리 약화, 매크로 불확실성에서 언급되며,  
구조적으로는 수급 불균형 심화가 주요 변수로 제시된다.  
유진투자증권(2026-04-06)은 가격 변동성과 투자심리 약화를 중심으로 단기 리스크를 강조하고,  
대신증권(2026-04-06)은 수급 불균형 자체를 구조적 변수로 제시하며,  
SK증권(2026-04-08)은 장기공급계약을 통해 일부 리스크 완화 가능성을 함께 언급한다.

## 시장 분석

2026년 메모리 반도체 시장은 AI 수요 확대와 제한적 공급 증가가 동시에 나타나며 수급 불균형이 심화되는 상황으로 언급된다(대신증권, 2026-04-06).  
이 과정에서 메모리 가격 상승 기반의 이익 확대 구조가 형성되고 있으나, 일부 시장에서는 DDR5 현물가격이 약 30% 급락하는 등 단기 가격 변동성이 나타나고 있다(유진투자증권, 2026-04-06).  
또한 호르무즈 해협 관련 불확실성 등 매크로 요인이 투자심리 약화와 주가 변동성 확대의 배경으로 제시된다(유진투자증권, 2026-04-06).

## 증권사별 관점 차이

| 구분 | 유진투자증권 (2026-04-06) | 대신증권 (2026-04-06) | SK증권 (2026-04-08) |
|------|--------------------------|----------------------|----------------------|
| 리스크 초점 | 가격 변동성·투자심리 | 수급 불균형 | 안정성 및 밸류 반영 |
| 시계 | 단기 | 구조적 | 중장기 |
| 주요 언급 내용 | DDR5 가격, 매크로 변수 | AI 수요 vs 공급 | 장기계약, P/E 조정 |

유진투자증권은 DDR5 현물가격 약세와 투자심리 약화를 중심으로 단기 변동성에 초점을 맞춘다.  
대신증권은 AI 수요 확대와 공급 제약이 결합된 수급 불균형을 구조적 변수로 제시한다.  
SK증권은 장기공급계약을 통해 실적 안정성이 강화되는 측면과 함께, 밸류에이션 조정을 통해 리스크를 반영한다.

## 주요 논거 포인트

**단기 리스크 (3~6개월)**

**① DDR5 현물가격 변동성 (유진투자증권, 2026-04-06)**  
- 일부 시장에서 DDR5 현물가 약 **30% 급락**  
- 딜러 차익 실현 및 단기 수급 요인으로 가격 변동성 발생

**② 투자심리 및 매크로 요인 (유진투자증권, 2026-04-06)**  
- 호르무즈 불안 등 매크로 변수 → 투자심리 약화  
- 최근 주가 약세의 주요 배경으로 언급

**구조적 리스크 (1년 이상)**

**③ 수급 불균형 심화 (대신증권, 2026-04-06)**  
- AI 수요 확대 + 제한적 공급 증가 → 수급 불균형 지속  
- 가격 상승과 이익 확대 구조의 기반이지만 변동성 요인으로도 작용

**완화 요인**

**⑤ 장기공급계약 및 밸류 반영 (SK증권, 2026-04-08)**  
- 장기공급계약 가시화 → 실적 안정성 강화  
- Target P/E **13.0배 → 11.0배 하향 적용** → 리스크를 밸류에이션에 반영

## 추가 섹션
"""),

    # ── few-shot 6: consensus ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 공통 논거를 항목별로 수치와 함께 정리하고, 컨센서스 배경과 이견 영역을 구분
[질문] NAND 업황에 대해 증권사들이 공통으로 강조하는 게 뭐야
[발췌]
### [DS투자증권] (리포트 날짜: 2026-04-16)
NAND는 감산 효과 누적, 재고 정상화, eSSD 중심 수요 회복으로 수급 개선.
1Q26 NAND 영업이익률 53% 추정, 계약가격 QoQ +55~60% 상승.
2027년까지 수급 타이트 유지 전망.

### [SK증권] (리포트 날짜: 2026-04-15)
NAND Spot Price 2월말 이후 6주간 최대 80% 상승.
AI 추론 확대로 KV 캐시 수요 증가 → HBM에서 SSD로 수요 확산.
공급 측 제약으로 단기 증설 대응 어려움.

### [대신증권] (리포트 날짜: 2026-04-06)
AI 추론 확장에 따라 NAND 제품 다각화 본격화.
2026년 NAND ASP YoY +164% 상승 전망.
공급업계 최소 투자 기조 유지 → 가격 상승 요인."""),
("ai", """\
## 핵심 요약

세 증권사는 모두 NAND 업황에 대해 **수요 확대 + 공급 제약에 따른 가격 상승 구조**를 공통적으로 강조한다.
DS투자증권(2026-04-16)은 계약가격 QoQ **+55~60% 상승**과 2027년까지 수급 타이트를 제시했고,
SK증권(2026-04-15)은 Spot Price **6주간 최대 +80% 상승**을 근거로 공급 부족 심화를 강조한다.
대신증권(2026-04-06)은 NAND ASP가 2026년 **+164% 상승**할 것으로 전망하며 가격 상승 사이클을 명확히 제시한다.

## 시장 분석

2026년 NAND 시장은 AI 추론 확장에 따른 데이터 저장 수요 증가와 제한적인 공급 증가가 맞물리며 구조적 수급 개선 국면에 진입했다.
SK증권(2026-04-15)에 따르면 AI 추론 과정에서 KV 캐시 수요가 증가하며 수요가 HBM에서 SSD까지 확산되고 있고,
DS투자증권(2026-04-16)은 eSSD 중심의 데이터센터 수요 회복이 NAND 수급 개선을 견인하고 있다고 분석한다.
공급 측면에서는 대신증권(2026-04-06)이 지적한 바와 같이 최소 투자 기조가 유지되는 가운데,
증설 리드타임으로 인해 단기적인 공급 대응이 어려운 구조가 형성되어 있다.

## 증권사별 관점 차이

세 증권사 모두 NAND 업황 개선에 대해 동일한 방향성을 제시하고 있으며,  
관점 차이는 “강조 포인트” 수준에 국한된다.

| 구분 | DS투자증권 | SK증권 | 대신증권 |
|------|------------|--------|----------|
| 수요 핵심 | eSSD·데이터센터 | AI 추론 → SSD 확산 | AI 기반 제품 다각화 |
| 가격 지표 | 계약가격 QoQ +55~60% | Spot +80% | ASP YoY +164% |
| 공급 해석 | 감산·재고 정상화 | 증설 제약 | 최소 투자 기조 |

👉 결론적으로 방향성 차이는 없으며 **관점 차이 없음**에 해당한다.

## 주요 논거 포인트

**① AI 기반 수요 확장 (3사 공통)**
- SK증권(2026-04-15): KV 캐시 수요 증가 → SSD까지 수요 확산
- 대신증권(2026-04-06): AI 추론 확장 → NAND 제품 다각화
- DS투자증권(2026-04-16): 데이터센터·eSSD 수요 회복

→ AI 인프라 확장이 NAND 수요를 구조적으로 재평가

**② 공급 제약 구조 (3사 공통)**
- SK증권(2026-04-15): 단기 증설 불가, 물리적 투자 제약
- 대신증권(2026-04-06): 최소 투자 기조 유지
- DS투자증권(2026-04-16): 감산 효과 누적 + 재고 정상화

→ 공급은 수요 증가 속도를 따라가지 못하는 구조

**③ 가격 상승 사이클 진입 (3사 공통)**
- DS투자증권(2026-04-16): 계약가격 QoQ **+55~60%**
- SK증권(2026-04-15): Spot Price **6주간 +80%**
- 대신증권(2026-04-06): NAND ASP YoY **+164%**

→ 가격 지표 전반에서 상승 추세가 동시에 확인

**④ 컨센서스의 핵심 구조**

세 증권사의 공통 인식은 다음 구조로 정리된다:

- 수요: AI 추론 확장 → SSD/NAND 수요 증가  
- 공급: 투자 부족 + 증설 지연 → 공급 제약 지속  
- 결과: 가격 상승 → 이익 개선

즉, NAND는 DRAM에 후행하던 사이클에서 벗어나  
**독립적인 상승 사이클 초입에 진입했다는 점이 컨센서스다.**

## 컨센서스 외 이견 영역

수요·공급·가격 방향성에 대한 이견은 존재하지 않는다.

다만,
- SK증권은 단기 공급 부족과 증설 제약을,
- 대신증권은 구조적 ASP 상승과 제품 믹스를,
- DS투자증권은 수요 회복 속도를

각각 더 강조하고 있어 **강조 포인트의 차이만 존재한다.**

## 추가 섹션
"""),

# ── 실제 질문 ─────────────────────────────────────────────────────────
    ("human", """\
[구조 가이드] {structure_hint}
[질문] {question}
[발췌]
{context}"""),
])


def _generate_answer(question: str, context: str, structure_hint: str) -> str:
    chain = _ANSWER_PROMPT | _llm_strong()
    return chain.invoke({
        "question":       question,
        "context":        context,
        "structure_hint": structure_hint,
    }).content


# ─────────────────────────────────────────────────────────────────────────────
# 투자 조언 요청 차단
# ─────────────────────────────────────────────────────────────────────────────

_INVESTMENT_ADVICE_KEYWORDS = [
    "사야", "팔아야", "매수해야", "매도해야",
    "지금 투자", "투자하기 좋은", "어디에 투자", "사도 돼", "팔아도 돼", "살까", "팔까", "투자해도 돼", "투자해도 괜찮",
    "추천해줘", "추천 종목", "좋은 종목",
    "어느 종목", "어떤 종목", "종목 추천",
    "수익 낼 수 있는", "오를 것 같은", "내릴 것 같은",
]

def _is_investment_advice_request(question: str) -> bool:
    """투자 추천·매수매도 조언 요청 여부 판별"""
    return any(keyword in question for keyword in _INVESTMENT_ADVICE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 모호한 질문(other) 경로: 5스텝 풀 파이프라인을 내장
# ─────────────────────────────────────────────────────────────────────────────

# ── Step A: 증권사별 요약 ────────────────────────────────────────────────────

_SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
당신은 15년 경력의 금융 리서치 전문 애널리스트입니다.

아래는 {broker}에서 발행한 리서치 리포트 원문입니다.
분석 주제: {topic}

[리포트 내용]
{content}

이 리포트를 단순 요약하지 말고, 최종 종합 리포트의 기초 재료로 사용할 수 있도록
**데이터 → 해석 → 논거 → 리스크** 구조로 정리하세요.

## 절대 원칙
- 리포트에 명시된 수치·사실만 사용하세요.
- 모든 핵심 수치에는 가능한 경우 리포트 날짜와 증권사명을 함께 남기세요.
- 발췌문에 없는 내용은 절대 생성하지 말고 "리포트에서 확인되지 않음"이라고 쓰세요.
- 일반론이나 산업 상식이 아니라, 이 리포트에만 있는 고유한 판단을 중심으로 작성하세요.
- 발췌문에 없는 인과관계·시장 함의·전망을 추론하거나 창작하지 마세요.
- 해석이 필요한 경우 반드시 발췌문에 있는 수치·문장에 근거한 "발췌문 기반 해석"으로만 제한하세요.

---

## 1. 핵심 수치 및 사실 스냅샷

이 리포트에서 확인되는 핵심 수치와 사실을 정리하세요.

포함 대상:
- 투자의견
- 목표주가
- 실적 추정치
- 성장률
- 가격 변화
- 수급 지표
- 밸류에이션 배수
- 리포트 날짜
- 주요 이벤트

※ 이 섹션은 해석을 최소화하고 fact 중심으로 작성하세요.

---

## 2. 이 증권사의 핵심 결론

이 증권사가 {topic}에 대해 내린 핵심 판단을 3~5문장으로 정리하세요.

반드시 포함:
- 긍정 / 중립 / 부정 중 어느 쪽에 가까운지
- 그 판단의 가장 중요한 근거
- 결론을 좌우하는 핵심 변수

---

## 3. 핵심 논거 구조

이 리포트의 주장을 뒷받침하는 논거를 정리하세요.

각 논거는 아래 구조로 작성하세요.

① 논거명  
- 근거 데이터:
- 해석:
- 이 논거가 결론으로 이어지는 방식:

단순 나열하지 말고, “왜 이 데이터가 이 결론으로 이어지는지”를 설명하세요.

---

## 4. 시장 변화 및 트리거

리포트 안에서 시점 변화, 업황 변화, 의견 변화가 확인되면 정리하세요.

반드시 확인할 것:
- 이전 상황
- 현재 상황
- 변화의 트리거
- 변화 이후 증권사의 해석

변화가 명확하지 않으면 "리포트에서 명확한 시계열 변화는 확인되지 않음"이라고 쓰세요.

---

## 5. 리스크 요인

리포트에서 언급된 리스크를 정리하세요.

구분:
- 단기 리스크
- 구조적 리스크
- 발생 조건
- 실적·주가·업황에 미치는 영향 경로

리스크가 언급되지 않았다면 "리포트에서 명시적 리스크 요인은 확인되지 않음"이라고 쓰세요.

---

## 6. 차별화된 시각

이 증권사가 다른 증권사와 다르게 볼 가능성이 있는 지점을 정리하세요.

포함 기준:
- 다른 데이터를 중시하는가
- 같은 데이터를 다르게 해석하는가
- 단기보다 중장기를 보는가, 또는 반대인가
- 밸류에이션이나 실적 가정이 독특한가

※ 타 증권사와 비교할 수 없는 경우, 또는 발췌문에서 차별화 근거를 확인할 수 없는 경우에는
  이 섹션을 생략하거나 "수집된 발췌문에서 차별화된 시각을 확인할 수 없음"이라고 명시하세요.
  발췌문에 없는 내용을 추론하거나 창작하지 마세요.
""")

def _summarize_by_broker(docs: list[Document], topic: str) -> dict[str, str]:
    print("  → 증권사별 요약 생성 중...")
    # ✅ 증권사별 요약은 8블록 최종 리포트의 기초 재료이므로 품질이 중요 → gpt-4o 사용
    llm = _llm_strong()

    broker_chunks: dict[str, list[str]] = {}
    broker_titles: dict[str, list[str]] = {}

    for doc in docs:
        broker = _get_firm(doc)
        title  = doc.metadata.get("title", "")
        broker_chunks.setdefault(broker, []).append(doc.page_content)
        if title and title not in broker_titles.get(broker, []):
            broker_titles.setdefault(broker, []).append(title)

    summaries = {}
    chain = _SUMMARY_PROMPT | llm

    for broker, chunks in broker_chunks.items():
        print(f"     · {broker} 요약 중...")
        # ✅ 청크 수에 따라 예산 균등 배분 — 단순 joined[:6000] 은 앞쪽 청크만 살아남는 문제 방지
        budget    = 6000
        per_chunk = max(budget // len(chunks), 300)  # 최소 300자 보장
        content   = "\n\n---\n\n".join(c[:per_chunk] for c in chunks)
        titles  = ", ".join(broker_titles.get(broker, [])[:3])
        response = chain.invoke({
            "broker":  broker,
            "topic":   topic,
            "content": f"[리포트 제목: {titles}]\n\n{content}",
        })
        summaries[broker] = response.content

    print(f"  → {len(summaries)}개 증권사 요약 완료")
    return summaries


# ── Step B-1: 컨센서스 분석 ────────────────────────────────────────────────

_CONSENSUS_PROMPT = ChatPromptTemplate.from_template("""
당신은 기관 투자자 대상 시장 리서치 전문가입니다.

아래는 '{topic}'에 대한 여러 증권사 리포트 요약입니다.
각 증권사의 주장을 비교하여, 최종 리포트의
**공통 인식 / 핵심 이견 / 커버리지 구조** 섹션에 들어갈 분석을 작성하세요.

[증권사별 분석 요약]
{summaries}

---

## 절대 원칙
- 제공된 요약에 있는 내용만 사용하세요.
- 수치가 있으면 반드시 증권사명과 함께 인용하세요.
- 단순 나열하지 말고, 왜 그런 공통 인식이 형성됐는지 설명하세요.
- 근거가 부족한 내용은 "요약 자료에서 확인되지 않음"이라고 명시하세요.

---

## 1. 커버리지 스냅샷

증권사들이 {topic}을 어떤 방식으로 다루고 있는지 정리하세요.

포함:
- 어떤 증권사가 참여했는가
- 심층 분석인지, 이벤트 코멘트인지, 위클리성 언급인지
- 커버리지의 깊이 차이
- 분석 초점의 차이

---

## 2. 시장 컨센서스

대다수 증권사가 공통으로 동의하는 내용을 정리하세요.

반드시 포함:
- 공통 결론
- 공통으로 사용하는 핵심 데이터
- 컨센서스가 형성된 구조적 배경
- 공통 논거가 어떤 결론으로 이어지는지

형식:
① 공통 인식  
- 참여 증권사:
- 근거 데이터:
- 구조적 배경:
- 해석:

---

## 3. 컨센서스의 강도

컨센서스가 강한지 약한지 판단하세요.

구분:
- 강한 컨센서스: 증권사들이 같은 데이터와 같은 해석을 공유
- 약한 컨센서스: 방향은 같지만 근거와 시간축이 다름
- 컨센서스 부재: 증권사별 판단이 엇갈림

왜 그렇게 판단했는지 설명하세요.

---

## 4. 아직 합의되지 않은 영역

증권사들이 명확히 합의하지 못한 부분을 정리하세요.

※ 발췌문에서 명확히 엇갈린 내용만 기술하세요.
  이견이 없는 경우 “수집된 발췌문에서 유의미한 이견 없음”이라고 명시하세요.
  항목을 채우기 위해 발췌문에 없는 이견을 만들지 마세요.

각 이견은 “무엇이 다르고, 왜 다른가”까지 설명하세요.
""")

# ── Step B-2: 차이점 분석 ─────────────────────────────────────────────────

_DIFFERENCE_PROMPT = ChatPromptTemplate.from_template("""
당신은 시장 리서치 전문가입니다.

아래는 '{topic}'에 대한 증권사별 분석 요약입니다.
같은 현상이나 데이터를 증권사들이 어떻게 다르게 해석하는지 분석하세요.

[증권사별 분석 요약]
{summaries}

---

## 절대 원칙
- 증권사 간 차이를 단순 나열하지 마세요.
- 반드시 “왜 다르게 해석했는가”를 설명하세요.
- 제공된 요약에 없는 내용은 추정하지 마세요.
- 수치와 사실에는 증권사명을 함께 표시하세요.

---

## 1. 핵심 이견 요약

가장 중요한 이견을 최대 5개까지 선정하세요.

※ 이견이 적으면 적은 개수만 작성하세요.
※ 이견이 없는 경우 "유의미한 이견 없음"이라고 명시하세요.

각 이견은 아래 구조로 작성하세요.

① 이견 주제  
- A 증권사 시각:
- B 증권사 시각:
- 근거 데이터:
- 차이가 발생한 이유:
- 분석 판단상 의미 (자료 내 근거가 있는 경우에만):

---

## 2. 데이터 선택 차이

증권사들이 어떤 데이터를 더 중요하게 보는지 비교하세요.

예:
- 수출 데이터
- 매출 데이터
- 가격 지표
- 수급 지표
- 실적 추정치
- 밸류에이션 배수
- 정책·이벤트 변수

단순히 “다른 데이터를 봤다”가 아니라,
그 데이터 선택이 결론을 어떻게 바꾸는지 설명하세요.

---

## 3. 해석 프레임 차이

같은 데이터를 두고 결론이 달라지는 경우를 분석하세요.

반드시 포함:
- 같은 데이터
- 서로 다른 해석
- 해석 차이의 원인
- 자료에서 한쪽 해석을 지지하는 추가 근거가 확인되면 기술 (없으면 "판단 근거 확인되지 않음"으로 명시)

---

## 4. 시간축 차이

증권사들이 단기와 중장기 중 어디에 초점을 두는지 분석하세요.

구분:
- 단기 실적·가격 모멘텀 중심
- 중장기 구조 변화 중심
- 단기 리스크 vs 장기 성장성 충돌

시간축 차이가 최종 결론을 어떻게 바꾸는지 설명하세요.

---

## 5. 이견을 해소할 핵심 변수

증권사 요약 자료에서 이견 해소와 관련해 직접 언급된 변수만 정리하세요.
자료에서 명시되지 않은 향후 데이터·이벤트를 새로 만들지 마세요.
해당 변수가 없으면 "자료에서 이견 해소 변수가 명시적으로 확인되지 않음"이라고 쓰세요.

확인된 변수별로:
- 어떤 증권사가 언급했는가 (출처 표기)
- 왜 중요한가
- 긍정적으로 나오면 어떤 해석이 강화되는가
- 부정적으로 나오면 어떤 해석이 약화되는가
""")

def _analyze_consensus(summaries: dict[str, str], topic: str) -> tuple[str, str]:
    print("  → 컨센서스 & 차이점 분석 중...")
    # ✅ summaries는 gpt-4o로 생성됐으므로, 이를 분석하는 단계도 동일 모델로 품질 일관성 유지
    llm = _llm_strong()
    summaries_text = "\n\n".join([f"[{b}]\n{s}" for b, s in summaries.items()])

    # ✅ 두 LLM 호출을 병렬 실행해 지연 시간을 절반으로 단축
    with ThreadPoolExecutor(max_workers=2) as ex:
        fc = ex.submit(
            (_CONSENSUS_PROMPT  | llm).invoke,
            {"topic": topic, "summaries": summaries_text},
        )
        fd = ex.submit(
            (_DIFFERENCE_PROMPT | llm).invoke,
            {"topic": topic, "summaries": summaries_text},
        )
        consensus_result   = fc.result()
        differences_result = fd.result()

    print(" → 분석 완료")
    return consensus_result.content, differences_result.content


# ── Step C: 인사이트 도출 ────────────────────────────────────────────────────

_INSIGHT_PROMPT = ChatPromptTemplate.from_template("""
당신은 20년 경력의 시니어 리서치 애널리스트이자 포트폴리오 매니저입니다.

아래 자료를 바탕으로 '{topic}'에 대한 최종 종합 리포트에 들어갈
**핵심 인사이트, 전망, 모니터링 변수**를 도출하세요.

주제: {topic}

[증권사별 분석 요약]
{summaries}

[컨센서스 분석]
{consensus}

[이견 분석]
{differences}

---

## 절대 원칙
- 제공된 자료에 근거한 인사이트만 작성하세요.
- 발췌문에 없는 사실을 새로 만들지 마세요.
- 단순 요약이 아니라 “데이터 간 연결”과 “해석의 함의”를 도출하세요.
- 투자 전략 제언, 매수·매도 권고, 포지션 비중 제안은 작성하지 마세요.
- 모든 인사이트는 “근거 데이터 → 해석 → 함의” 구조로 작성하세요.

---

## 1. Top 5 핵심 인사이트

가장 중요한 인사이트 5개를 작성하세요.

각 인사이트는 아래 구조를 따르세요.

① 인사이트 제목  
- 근거 데이터:
- 해석:
- 최종 리포트에서 연결될 섹션:
※ 발췌문에 명시되지 않은 내용("시장이 놓치기 쉬운 부분" 등)은 추론하거나 창작하지 마세요.

주의:
- 이미 컨센서스에 나온 내용을 반복하지 마세요.
- 서로 다른 증권사 자료를 연결해 더 깊은 함의를 도출하세요.
- “좋다/나쁘다”가 아니라 “왜 중요한가”를 설명하세요.

---

## 2. 발췌문에서 충분히 다뤄지지 않은 변수

발췌문에 언급됐으나 다른 섹션에서 아직 다뤄지지 않은 변수만 정리하세요.

각 변수별로:
- 발췌문에서 어떻게 언급됐는가 (출처 증권사·날짜 포함)
- 왜 추가 주목이 필요한가

※ 발췌문에 없는 변수를 새로 창작하지 마세요.
  해당하는 변수가 없으면 이 섹션을 생략하세요.

---

## 3. 단기 vs 중장기 전망의 괴리

3~6개월 단기 전망과 1~2년 중장기 전망이 자료에서 다르게 나타나는 부분을 분석하세요.
자료에서 시간축 차이가 확인되지 않으면 이 섹션을 생략하거나 "시간축 괴리 확인되지 않음"이라고 쓰세요.

확인된 경우 아래 항목 중 해당하는 것만 작성하세요 (해당 없으면 생략):
- 단기에는 긍정적이나 중장기에는 리스크인 요소
- 단기에는 부담이나 중장기에는 기회인 요소
- 증권사별 시간축 차이
- 전망 전환의 트리거가 자료에서 명시된 경우

---

## 4. 시나리오 구조

Bull / Base / Bear 시나리오는 **제공된 자료에서 조건·변수·논거가 명시적으로 확인되는 경우에만** 작성하세요.
항목을 채우기 위해 발췌문에 없는 조건이나 데이터를 만들지 마세요.

작성 기준:
- Bull Case: 상승 조건 또는 긍정 시나리오가 자료에 명시된 경우에만 작성
- Base Case: 자료가 가장 많이 지지하는 기본 시나리오가 확인되는 경우에만 작성
- Bear Case: 하방 변수·리스크·깨지는 조건이 자료에 명시된 경우에만 작성
- 해당 근거가 없으면 해당 Case 전체를 생략하거나 "자료에서 명시적 조건 확인되지 않음"이라고 쓰세요.

각 시나리오는 아래 구조를 따르되, 근거가 확인되지 않는 하위 항목은 "자료에서 확인되지 않음"으로 표시하세요.

### Bull Case
- 성립 조건:
- 강화되는 증권사 논리:
- 확인해야 할 데이터:

### Base Case
- 현재 자료가 가장 많이 지지하는 기본 시나리오:
- 근거:
- 남아 있는 불확실성:

### Bear Case
- 깨지는 변수:
- 현실화 조건:
- 영향을 받는 논거:

---

## 5. 핵심 모니터링 변수 Top 5

앞으로 가장 중요하게 봐야 할 데이터·이벤트·지표 5개를 선정하세요.

각 변수별로:
- 변수명
- 자료에서 언급한 출처 (증권사, 날짜)
- 왜 중요한가 (자료 내 근거)
- 자료에서 해석 방향이 제시된 경우에만: 긍정/부정 방향
- 연결되는 증권사 논거

주의:
- 투자 행동 권고는 쓰지 마세요.
- 최종 리포트의 “핵심 모니터링 변수” 섹션에 바로 사용할 수 있는 형태로 작성하세요.
""")

def _extract_insights(
    summaries:   dict[str, str],
    consensus:   str,
    differences: str,
    topic:       str,
) -> str:
    print("  → 핵심 인사이트 도출 중...")
    llm = _llm_strong()
    summaries_text = "\n\n".join([f"[{b}]\n{s}" for b, s in summaries.items()])
    response = (_INSIGHT_PROMPT | llm).invoke({
        "topic":       topic,
        "summaries":   summaries_text,
        "consensus":   consensus,
        "differences": differences,
    })
    print("  → 완료")
    return response.content


# ── Step D: 최종 8블록 리포트 생성 ───────────────────────────────────────────

_FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template("""
당신은 15년 이상 경력의 기관 투자자 대상 리서치 애널리스트입니다.

이 작업은 단순 요약이 아니라  
**여러 증권사 리포트를 기반으로 “데이터 → 해석 → 비교 → 판단”의 추론 과정을 수행하는 고급 분석 작업입니다.**

---

## ⚠️ 절대 원칙

- 제공된 분석 자료와 원문 발췌 근거에 포함된 **구체적 수치만 인용**
- 모든 수치에는 반드시 **출처(증권사, 날짜)** 표기
- 제공된 분석 자료와 원문 발췌 근거에 없는 정보는 절대 생성 금지 → "리포트에서 확인되지 않음" 명시
- 일반론 / 산업 상식 / 교과서 설명 금지
- 모든 문장은 반드시 **“데이터 → 해석” 구조를 포함**
- 단, 0번 섹션(데이터 및 커버리지 스냅샷)은 해석 없이 fact만 작성
- 증권사 간 차이는 반드시 **“왜 다르게 해석했는가”까지 분석**
- 중간 분석과 원문 발췌 근거가 충돌하면 반드시 원문 발췌 근거를 우선하고, 불일치는 "원문 근거 기준으로 재확인 필요"라고 명시

---

## 🎯 리포트 목표

이 리포트는 반드시 다음을 달성해야 한다:

1. 시장에서 실제로 벌어지고 있는 변화의 구조 설명
2. 동일 데이터를 두고 증권사들이 왜 다른 결론을 내리는지 해석
3. 투자 판단에 영향을 주는 핵심 변수 식별

---

## 🔴 리포트 작성 순서 (반드시 준수)

이 리포트는 아래 순서를 반드시 따른다:

1. 데이터 및 커버리지 스냅샷 (fact only)
2. 핵심 요약
3. 시장 변화 및 흐름 (timeline)
4. 증권사 관점 비교 (core reasoning)
5. 핵심 논거 구조 (thesis)
6. 리스크 구조 (risk framework)
7. 전망 (forward logic)
8. 핵심 모니터링 변수

※ 순서를 변경하면 잘못된 리포트로 간주한다

---

주제: {topic}  
작성일: {date}  
참고 증권사: {brokers}

---

[분석 자료]

=== 증권사별 핵심 분석 ===
{summaries}

=== 컨센서스 및 이견 ===
{consensus}

{differences}

=== 핵심 인사이트 ===
{insights}

=== 원문 발췌 근거 (검증용) ===
{source_context}

---

# {topic} 종합 리서치 리포트

---

## 0. 데이터 및 커버리지 스냅샷

- 이번 분석에 포함된 증권사 및 리포트 시점 정리
- 핵심 수치 (목표주가, 성장률, 실적 추정 등) 요약
- 이 섹션은 **해석 없이 fact만 정리**

---

## 1. 핵심 요약 (Executive Insight)

- 전체 분석을 5~7문장으로 압축
- 반드시 포함:
  - 가장 중요한 수치 1개
  - 가장 중요한 구조적 변화 1개
  - 가장 큰 증권사 간 관점 차이 1개

---

## 2. 시장 변화 및 흐름 (Timeline Analysis)

- 반드시 BEFORE → AFTER 구조로 설명
- 변화가 없다면 “변화 없음” 명시
- 변화의 원인(트리거)을 명확히 설명

---

## 3. 증권사 관점 비교 (Interpretation Gap Analysis)

### 3.1 공통 인식 (Consensus)
- 대부분 증권사가 동의하는 구조
- 왜 컨센서스가 형성됐는지 설명

### 3.2 핵심 이견 (Divergence)
- 의견이 갈리는 지점 명확히 정의
- 단순 긍정/부정 구분 금지

### 3.3 관점 차이의 근본 원인 (핵심)

발췌문에서 확인되는 차이만 아래 기준으로 분석하세요.
해당하는 항목이 없으면 "차이 확인되지 않음"이라고 명시하고 생략하세요.

1) 데이터 선택 차이  
2) 해석 프레임 차이  
3) 시간축 차이 (단기 vs 중장기)

---

## 4. 핵심 논거 구조 (Thesis Breakdown)

- 공통 논거 vs 증권사별 고유 논거 구분
- 논거 간 충돌 지점 분석
- 각 논거가 어떤 결론으로 이어지는지 설명

---

## 5. 리스크 구조 (Risk Framework)

### 5.1 단기 리스크
- 3~6개월 내 발생 가능
- 발생 조건과 영향 경로 설명

### 5.2 구조적 리스크
- 장기 투자 thesis에 영향을 주는 요소

### 5.3 시나리오 구조

- Bull Case → 원문 발췌 근거 또는 분석 자료에 상승 조건이 명시된 경우에만 작성  
- Base Case → 현재 자료가 가장 많이 지지하는 기본 시나리오가 확인되는 경우에만 작성  
- Bear Case → 원문 발췌 근거 또는 분석 자료에 하방 변수·리스크가 명시된 경우에만 작성  
- 근거가 확인되지 않는 Case는 생략하거나 "자료에서 명시적 조건 확인되지 않음"으로 표시  

---

## 6. 전망 (Forward View)

### 6.1 단기 전망 (3~6개월)
- 이벤트 기반 시나리오
- 어떤 데이터에 따라 방향이 바뀌는지

### 6.2 중장기 전망 (1~2년)
- 구조 변화가 어떤 결과로 이어지는지
- 자료에서 명시적으로 언급된 미반영 요소가 있으면 인용 (없으면 생략)

---

## 7. 핵심 모니터링 변수

- 분석 자료 또는 원문 발췌 근거에서 직접 언급된 변수 중 가장 중요한 것을 최대 5개 선정
- 자료에서 언급되지 않은 변수를 새로 만들지 마세요
- 각 변수에 대해:
  - 출처 (증권사, 날짜)
  - 왜 중요한가 (자료 내 근거)
  - 자료에서 해석 방향이 제시된 경우에만 방향성 기술

---

## ❗ 출력 스타일

- 마크다운 사용
- 불필요한 장식 금지
- 표는 꼭 필요한 경우만 사용
- 모든 문장은 “데이터 → 해석” 구조 유지

""")


def _build_source_context_for_report(docs: list[Document], max_total_chars: int = 12000) -> str:
    """최종 리포트 단계에서 중간 요약 오류를 교정할 수 있도록 원문 청크 일부를 함께 전달한다.

    증권사별/날짜별 메타데이터를 유지하고, 각 청크는 일정 길이까지만 제공해 토큰 폭증을 방지한다.
    """
    if not docs:
        return "원문 발췌 근거 없음"

    parts: list[str] = []
    used = 0
    per_doc_limit = 1200

    for i, doc in enumerate(docs, 1):
        firm = _get_firm(doc)
        date = doc.metadata.get("report_date", "날짜 미상") or "날짜 미상"
        title = doc.metadata.get("title", "") or ""
        filename = doc.metadata.get("filename", "") or ""
        body = doc.page_content.strip()[:per_doc_limit]
        header = f"### [{i}] {firm} / {date}"
        if title:
            header += f" / {title}"
        elif filename:
            header += f" / {filename}"
        block = f"{header}\n{body}"

        if used + len(block) > max_total_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts) if parts else "원문 발췌 근거 없음"


def _generate_full_report(
    topic:       str,
    summaries:   dict[str, str],
    consensus:   str,
    differences: str,
    insights:    str,
    source_context: str,
) -> str:
    print("  → 최종 8블록 리포트 생성 중... (gpt-4o)")
    llm = _llm_strong()

    summaries_text = "\n\n".join([f"[{b}]\n{s}" for b, s in summaries.items()])
    brokers = ", ".join(summaries.keys())
    date    = datetime.now().strftime("%Y년 %m월 %d일")

    response = (_FINAL_REPORT_PROMPT | llm).invoke({
        "topic":       topic,
        "date":        date,
        "brokers":     brokers,
        "summaries":   summaries_text,
        "consensus":   consensus,
        "differences": differences,
        "insights":    insights,
        "source_context": source_context,
    })
    print("  → 완료")
    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# 면책조항(Disclaimer) 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_disclaimer(
    sources:        list[str],
    question_type:  str,
    mode:           str,
) -> str:
    """
    리포트 하단에 삽입할 면책조항 마크다운 블록을 반환한다.

    Args:
        sources:       참고 증권사 목록 (예: ["iM증권", "키움증권"])
        question_type: 질문 유형 (fact_lookup / coverage_summary / … / other)
        mode:          처리 모드 ("freeform" | "full_report")

    Returns:
        마크다운 형식의 면책조항 문자열
    """
    date_str     = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    sources_str  = ", ".join(sources) if sources else "알 수 없음"
    mode_label   = "종합 리포트 (풀 파이프라인)" if mode == "full_report" else f"요약 분석 ({question_type})"

    return f"""
---

> **⚠️ 면책조항 (Disclaimer)**
>
> 본 자료는 **{sources_str}** 에서 발행한 리서치 리포트를 검색·발췌하여 AI가 재구성한 참고용 문서입니다.
> 생성 시각: {date_str} | 분석 유형: {mode_label}
>
> - 본 자료는 **투자 권유, 매매 추천, 종목 추천을 목적으로 하지 않습니다.**
> - 본 자료에 인용된 수치·전망·투자의견은 **원본 리포트 발행 시점 기준**이며, 현재 시점과 다를 수 있습니다.
> - AI 검색·요약 과정에서 원문의 맥락 누락, 수치 오인용, 의미 변형이 발생할 수 있으므로 반드시 **원본 리포트를 직접 확인**하시기 바랍니다.
> - 동일 주제에 대해 수집되지 않은 증권사 리포트가 존재할 수 있으며, 본 자료는 수집된 리포트 범위 내에서만 유효합니다.
> - 투자 판단의 최종 책임은 이용자 본인에게 있습니다.
> - 본 서비스는 「자본시장과 금융투자업에 관한 법률」상 투자자문업 또는 투자일임업에 해당하지 않습니다.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(
    retrievers,
    question:    str,
    retrieve_fn: callable | None = None,
    rerank_fn:   callable | None = None,
    k_per_query: int  = 15,
    top_n:       int  = 12,
    # other 경로(풀 리포트)에서 사용할 검색 파라미터
    k_full:      int  = 20,
    top_n_full:  int  = 10,
    output_dir:  str  = "./data/reports_output",
    save:        bool = True,
) -> dict:
    """
    자유형 질문 → 유형 분류 → 분기:
      - 명확 유형(fact_lookup, coverage_summary, timeline, broker_comparison,
        risk, consensus) → 공통 4섹션 + 유형별 추가 구조(few-shot)
      - other(모호·복합 질문) → 증권사별 요약/컨센서스/이견/인사이트 기반
        8블록 종합 리포트

    Returns:
        {
            "question":      원본 질문,
            "question_type": 분류된 유형,
            "answer":        생성된 리포트 (마크다운),
            "sources":       참고 증권사 목록,
            "chunk_count":   사용한 청크 수,
            "mode":          "freeform" | "full_report",
        }
    """
    print("\n" + "=" * 60)
    print(f"입력: {question}")
    print("=" * 60)

    # ── 투자 조언 요청 사전 차단 (Step 1 이전, LLM 호출 없음) ─────────────
    if _is_investment_advice_request(question):
        blocked_answer = (
            "본 시스템은 투자 추천, 매수·매도 조언, 종목 추천을 제공하지 않습니다.\n\n"
            "증권사 리포트에 기반한 분석 질문으로 바꿔서 물어봐 주세요.\n\n"
            "예시:\n"
            "- \'반도체 섹터에 대한 증권사별 관점 차이는?\'\n"
            "- \'최근 조선업 리스크 요인을 정리해줘\'\n"
            "- \'하나증권과 키움증권의 반도체 의견 차이는?\'"
        )
        print("  → 투자 조언 요청 감지 → 차단")
        return {
            "question":      question,
            "question_type": "blocked",
            "answer":        blocked_answer,
            "sources":       [],
            "chunk_count":   0,
            "mode":          "blocked",
        }


    # ── Step 1: 질문 유형 분류 + 검색 쿼리 생성 ───────────────────────────
    print("\n[Step 1] 질문 유형 분류 및 검색 쿼리 생성 중...")
    intent = _analyze_intent(question)
    print(f"  → 유형: {intent['question_type']}")
    print(f"  → 섹터: {intent.get('target_sector') or '전체'}")
    print(f"  → 기간: {intent.get('target_period') or '전체'}")
    print(f"  → 대상 증권사: {intent['target_brokers'] or '전체'}")
    print(f"  → 검색 쿼리: {intent['search_queries']}")

    is_other = intent["question_type"] == "other"
    mode = "full_report" if is_other else "freeform"
    print(f"  → 처리 모드: {mode}")

    # ── Step 2: 다중 쿼리 검색 + 중복 제거 + Rerank ─────────────────────
    print("\n[Step 2] 리포트 청크 수집 및 rerank 중...")

    _intent = "balanced" if (
    intent["question_type"] in ("broker_comparison", "consensus", "other")
    or (intent["question_type"] == "coverage_summary" and len(intent["target_brokers"]) >= 2)
    ) else "ensemble"

    docs = _collect_chunks(
        retrievers,
        queries        = intent["search_queries"],
        target_brokers = intent["target_brokers"],
        target_period  = intent.get("target_period"),   
        target_sector  = intent.get("target_sector"),   
        retrieve_fn    = retrieve_fn,
        rerank_fn      = rerank_fn,
        k_per_query    = k_full if is_other else k_per_query,
        top_n          = top_n_full if is_other else top_n,
        intent         = _intent,
    )
    print(f"  → {len(docs)}개 청크 확보")

    if not docs:
        return {
            "question":      question,
            "question_type": intent["question_type"],
            "answer":        "관련 리포트를 찾을 수 없습니다.",
            "sources":       [],
            "chunk_count":   0,
            "mode":          mode,
        }

    # ── 요청한 증권사/섹터가 실제 검색 결과에 있는지 검증 ───────────────
    # 커버하지 않는 증권사/섹터에 대해 other → full_report로 우회 생성되는 것을 방지한다.
    docs, scope_error = _validate_and_filter_requested_scope(docs, intent)
    if scope_error:
        print(f"  → 요청 범위 검증 실패: {scope_error}")
        return {
            "question":      question,
            "question_type": intent["question_type"],
            "answer":        scope_error,
            "sources":       [],
            "chunk_count":   0,
            "mode":          "not_found",
        }

    sources = sorted({_get_firm(d) for d in docs})
    print(f"  → 참고 증권사: {sources}")

    # ── Step 3~4: 모드별 컨텍스트/리포트 생성 ───────────────────────────
    if is_other:
        # other 경로: 프롬프트 기준 풀 리포트 생성
        # Step 3에서 중간 분석(증권사별 요약 → 컨센서스/이견 → 인사이트)을 만들고,
        # Step 4에서 최종 8블록 종합 리포트를 생성한다.
        topic = intent.get("target_sector") or question

        print(f"\n[Step 3] 풀 리포트용 중간 분석 생성 중... (topic='{topic}')")
        summaries              = _summarize_by_broker(docs, topic)
        consensus, differences = _analyze_consensus(summaries, topic)
        insights               = _extract_insights(summaries, consensus, differences, topic)

        print("\n[Step 4] 8블록 종합 리포트 생성 중... (gpt-4o)")
        source_context         = _build_source_context_for_report(docs)
        answer                 = _generate_full_report(
            topic, summaries, consensus, differences, insights, source_context
        )
        print("  → 완료")
    else:
        print("\n[Step 3] freeform 답변용 증권사별 컨텍스트 구성 중...")
        context = _build_context(docs, intent["target_brokers"])

        print("\n[Step 4] 리포트 생성 중... (gpt-4o)")
        answer = _generate_answer(
            question       = question,
            context        = context,
            structure_hint = intent.get("structure_hint", "질문에 맞게 자유롭게 구성"),
        )
        print("  → 완료")

    # ── 면책조항 추가 ─────────────────────────────────────────────────────
    disclaimer = build_disclaimer(
        sources       = sources,
        question_type = intent["question_type"],
        mode          = mode,
    )
    answer = answer + disclaimer

    # ── 저장 ──────────────────────────────────────────────────────────
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        safe   = re.sub(r'[\\/:*?"<>|]', "_", question[:40])
        ts     = datetime.now().strftime("%Y%m%d_%H%M")
        prefix = "fullreport" if is_other else "freeform"
        base   = Path(output_dir) / f"{prefix}_{safe}_{ts}"

        if is_other:
            # 풀 리포트는 자체 헤더가 있어 그대로 저장
            (base.with_suffix(".md")).write_text(answer, encoding="utf-8")
        else:
            header = f"# Q: {question}\n\n> 참고 증권사: {', '.join(sources)}\n\n---\n\n"
            (base.with_suffix(".md")).write_text(header + answer, encoding="utf-8")

        sources_data = [
            {"content": d.page_content, **d.metadata} for d in docs
        ]
        (base.with_name(base.name + "_sources.json")).write_text(
            json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  저장 완료: {base.with_suffix('.md')}")

    print("=" * 60)

    return {
        "question":      question,
        "question_type": intent["question_type"],
        "answer":        answer,
        "sources":       sources,
        "chunk_count":   len(docs),
        "mode":          mode,
        "docs":          docs,
    }