"""
src/reportcreator/freeform_chain.py
자유형 질문 → 리포트 생성 체인 — 메인 진입점 (answer_question)

이 파일은 원래 2,181줄짜리 단일 파일이었으나, 역할별로 아래처럼 분리했습니다:
  - common.py          : 섹터/증권사 정규화, LLM 인스턴스, 메타데이터 헬퍼
  - intent.py           : Step 1 — 질문 유형 분류 + 검색 쿼리 생성 (7유형 few-shot)
  - retrieval.py         : Step 2 — 다중 쿼리 검색 + 중복 제거 + Rerank
  - freeform_answer.py   : Step 3~4 — 컨텍스트 구성 + 명확 유형 답변 생성 (few-shot)
  - full_report.py       : 모호한 질문(other) 경로 — 5스텝 풀 리포트 파이프라인
  - freeform_chain.py(현재 파일) : 위 모듈을 조립해서 answer_question() 으로 노출

기존에 이 파일에서 import 하던 코드(api/main.py, app.py, pipeline/main.py)는
그대로 동작합니다 — answer_question / _analyze_intent / _collect_chunks 를
이 파일에서 계속 re-export 하기 때문입니다.

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

    result = answer_question(retriever, "하나증권과 키움증권의 3월 의견 차이를 설명해줘", ...)
    result = answer_question(retriever, "반도체 섹터 어때?", ...)
    print(result["answer"])
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .common import _get_firm
from .intent import _analyze_intent, _validate_and_filter_requested_scope
from .retrieval import _collect_chunks
from .freeform_answer import _build_context, _generate_answer
from .full_report import (
    _is_investment_advice_request,
    _build_source_context_for_report,
    _summarize_by_broker,
    _analyze_consensus,
    _extract_insights,
    _generate_full_report,
    build_disclaimer,
)

__all__ = ["answer_question", "_analyze_intent", "_collect_chunks"]




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