"""
src/reportcreator/intent.py
Step 1: 질문 유형 분류 + 검색 쿼리 생성 (7유형 few-shot)

freeform_chain.py 리팩터링(2,181줄 → 역할별 분리)으로 만들어진 파일입니다.
로직은 원본에서 그대로 옮겨온 것이며 변경되지 않았습니다.
"""
from __future__ import annotations

import json
import re

from langchain.schema import Document, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate

from src.retriever.router import normalize_firms
from .common import _llm_fast, _get_firm


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


