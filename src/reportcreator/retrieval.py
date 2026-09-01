"""
src/reportcreator/retrieval.py
Step 2: 다중 쿼리 검색 + 중복 제거 + Rerank

주의: retrieve/rerank 함수는 main.py 가 retrieve_fn / rerank_fn 인자로 주입한다.
여기서 특정 전략을 직접 import 하면 main.py 의 전략 교체(RETRIEVER / RERANKER)가 무력화되므로,
이 모듈은 어떤 retriever/reranker 전략과도 결합 가능하도록 함수 인자에만 의존한다.
(아래 import 는 인자가 주어지지 않았을 때 쓰는 기본값(fallback) 용도)

freeform_chain.py 리팩터링(2,181줄 → 역할별 분리)으로 만들어진 파일입니다.
로직은 원본에서 그대로 옮겨온 것이며 변경되지 않았습니다.
"""
from __future__ import annotations

from langchain.schema import Document

from src.retriever.router import select_and_retrieve as _router_select_and_retrieve
from src.reranker.reranker_01_crossencoder import rerank as _default_rerank
from .common import normalize_sector, _get_firm


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
