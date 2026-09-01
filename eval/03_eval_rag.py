"""
eval/03_eval_rag.py
RAGAS 4 Metrics 로 RAG 파이프라인 평가

평가 지표:
  - context_precision : 관련 문서가 상위에 잘 배치됐는가?  (Retriever 정확도)
  - context_recall    : 필요한 정보를 빠짐없이 검색했는가? (Retriever 재현율)
  - faithfulness      : 답변이 컨텍스트에만 근거하는가?    (Hallucination 체크)
  - answer_relevancy  : 답변이 질문과 관련 있는가?         (생성 품질)

점수 기준:
  - 0.8 이상  : 양호 ✅
  - 0.6 ~ 0.8 : 개선 필요 ⚠️
  - 0.6 미만  : 문제 ❌

전제 조건:
  - eval/01_generate_testset.py 실행 완료 (data/eval/testset.csv 존재)
  - pipeline/ingest.py 실행 완료 (ChromaDB 존재)

★ 설정 블록 — 이곳만 수정 ★
  - CHUNKING, EMBEDDING, VECTORSTORE, RERANKER: 평가할 조합 선택
  - RAG_K, RAG_TOP_N: 검색/리랭크 파라미터

실행:
  uv run python eval/03_eval_rag.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ★ 설정 블록 — 평가할 조합 선택
# ══════════════════════════════════════════════════════════════════════════════

# ── 청킹 전략 (하나만 주석 해제) ──────────────────────────────────────────────
#from src.processing.chunking import chunking_01_recursive as CHUNKING
#from src.processing.chunking import chunking_02_semantic as CHUNKING
from src.processing.chunking import chunking_03_hybrid   as CHUNKING
# from src.processing.chunking import chunking_04_sentence as CHUNKING

# ── 임베딩 전략 ────────────────────────────────────────────────────────────────
from src.embedding import embedding_01_openai as EMBEDDING

# ── 벡터스토어 전략 ────────────────────────────────────────────────────────────
from src.vectorstore import vectorstore_01_chroma as VECTORSTORE

# ── 리랭커 전략 ────────────────────────────────────────────────────────────────
# HuggingFace 접속 가능한 환경: reranker_01_crossencoder

# from src.reranker import reranker_01_crossencoder as RERANKER
from src.reranker import reranker_01_crossencoder as RERANKER

# ── 라우터 (리트리버 자동 선택) ───────────────────────────────────────────────
from src.retriever import router as ROUTER

# ── 검색 파라미터 ─────────────────────────────────────────────────────────────
RAG_K     = 40   # Hybrid Search 후보 수 (Reranker 입력)
RAG_TOP_N = 10   # Reranker 통과 후 최종 컨텍스트 수

# ══════════════════════════════════════════════════════════════════════════════

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from eval.base import (
    get_evaluator_llm,
    get_evaluator_embeddings,
    load_testset,
    save_results,
    print_score_summary,
    RAG_EVAL_PATH,
)


# ── DB 경로 계산 ──────────────────────────────────────────────────────────────

def _get_db_path() -> str:
    """app.py 와 동일한 규칙으로 DB 경로 계산."""
    return str(
        BASE_DIR / "data" / "vectorstore"
        / VECTORSTORE.STRATEGY_NAME
        / EMBEDDING.STRATEGY_NAME
        / CHUNKING.STRATEGY_NAME
    )


# ── Retriever 준비 ────────────────────────────────────────────────────────────

def _load_vectorstore(db_path: str):
    """ChromaDB 로드 (DB 없으면 안내 후 종료)."""
    if not VECTORSTORE.exists(db_path):
        print(f"\n  ERROR: DB 없음 ({db_path})")
        print(f"  먼저 'uv run python pipeline/ingest.py' 를 실행하세요.")
        sys.exit(1)

    embeddings  = EMBEDDING.get_embeddings()
    vectorstore = VECTORSTORE.load(db_path, embeddings)
    return vectorstore


def _get_all_docs(vectorstore) -> list[Document]:
    """BM25 Retriever 용 Document 전체 로드."""
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]


# ── RAG 실행 ─────────────────────────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_template("""
당신은 금융 리포트 분석 전문가입니다.
아래 컨텍스트를 바탕으로 질문에 정확하게 답하세요.
컨텍스트에 없는 내용은 절대 추측하지 마세요.

[컨텍스트]
{context}

[질문]
{question}

[답변]
""")


def _collect_answers_and_contexts(
    questions: list[str],
    retrievers,          # ← router의 튜플 (ret1, ret2, all_docs, vectorstore)
) -> tuple[list[str], list[list[str]]]:
    """
    각 질문에 대해 RAG 실행 (production 파이프라인과 동일):
      1. Router 가 쿼리 의도 분석 후 리트리버 자동 선택
      2. Reranker 로 RAG_TOP_N개로 압축
      3. LLM 으로 답변 생성

    Returns:
      answers          : 생성된 답변 리스트
      retrieved_contexts: 검색된 컨텍스트 리스트 (각 질문별 문자열 리스트)
    """
    llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = _RAG_PROMPT | llm

    answers   = []
    retrieved = []

    print(f"  총 {len(questions)}개 질문 처리 중...")

    for i, question in enumerate(questions, 1):
        # router로 쿼리 의도 분석 후 리트리버 자동 선택 + 검색
        candidates    = ROUTER.retrieve(retrievers, question, k=RAG_K)
        docs          = RERANKER.rerank(question, candidates, top_n=RAG_TOP_N)
        context_texts = [d.page_content for d in docs]
        retrieved.append(context_texts)

        # 답변 생성
        context_str = "\n\n---\n\n".join(context_texts)
        response    = chain.invoke({"context": context_str, "question": question})
        answers.append(response.content)

        if i % 5 == 0 or i == len(questions):
            print(f"  [{i}/{len(questions)}] 완료")

    return answers, retrieved


# ── 평가 데이터셋 구성 ────────────────────────────────────────────────────────

def _build_eval_dataset(
    testset: Dataset,
    answers: list[str],
    retrieved_contexts: list[list[str]],
) -> Dataset:
    """
    RAGAS evaluate() 에 필요한 5개 컬럼 확인:
      - user_input          (테스트셋 생성 시 포함)
      - reference           (테스트셋 생성 시 포함)
      - reference_contexts  (테스트셋 생성 시 포함, 이미 리스트로 복원됨)
      - answer              (RAG 가 생성한 답변)     ← 지금 추가
      - retrieved_contexts  (Retriever 가 가져온 청크) ← 지금 추가
    """
    dataset = testset

    # RAGAS 0.4.x 기준 컬럼명: response (구버전의 answer 아님)
    for col in ("response", "answer", "retrieved_contexts"):
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    dataset = dataset.add_column("response", answers)
    dataset = dataset.add_column("retrieved_contexts", retrieved_contexts)
    return dataset


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    label = f"{CHUNKING.STRATEGY_NAME} + router + {RERANKER.STRATEGY_NAME}"

    print("=" * 65)
    print("RAG 파이프라인 평가 (RAGAS 4 Metrics)")
    print("=" * 65)
    print(f"  청킹     : {CHUNKING.STRATEGY_NAME}")
    print(f"  임베딩   : {EMBEDDING.STRATEGY_NAME}")
    print(f"  리트리버 : router (쿼리 의도에 따라 자동 선택)")
    print(f"  리랭커   : {RERANKER.STRATEGY_NAME}")
    print(f"  k={RAG_K} → rerank → top_n={RAG_TOP_N}")

    # STEP 1: 테스트셋 로드
    print("\n[STEP 1] 테스트셋 로드")
    testset   = load_testset()
    questions = testset["user_input"]
    print(f"  {len(questions)}개 Q&A 쌍 로드 완료")

    # STEP 2: Retriever 준비
    print("\n[STEP 2] Retriever / Reranker 준비")
    db_path     = _get_db_path()
    vectorstore = _load_vectorstore(db_path)
    all_docs    = _get_all_docs(vectorstore)
    retrievers  = ROUTER.build_retriever(vectorstore, all_docs, k=RAG_K)

    # STEP 3: RAG 실행
    print("\n[STEP 3] RAG 실행 (검색 + Rerank + 답변 생성)")
    answers, retrieved_contexts = _collect_answers_and_contexts(questions, retrievers)

    # STEP 4: 평가 데이터셋 구성
    print("\n[STEP 4] 평가 데이터셋 구성")
    eval_dataset = _build_eval_dataset(testset, answers, retrieved_contexts)
    print(f"  컬럼: {eval_dataset.column_names}")

    # STEP 5: RAGAS 평가
    print("\n[STEP 5] RAGAS 평가 실행...")
    print("  (LLM 으로 각 지표를 계산합니다 — 수 분 소요)")
    result = evaluate(
        dataset          = eval_dataset,
        metrics          = [context_precision, context_recall, faithfulness, answer_relevancy],
        llm              = get_evaluator_llm(),
        embeddings       = get_evaluator_embeddings(),
        raise_exceptions = False,   # 개별 샘플 실패 시 NaN 으로 처리 (전체 중단 방지)
    )

    # STEP 6: 결과 출력
    import math, numpy as np
    result_df_raw = result.to_pandas()

    print("\n  [DEBUG] 샘플별 원점수 (NaN 개수 확인):")
    metric_keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    for key in metric_keys:
        if key in result_df_raw.columns:
            col      = result_df_raw[key]
            nan_cnt  = col.isna().sum()
            mean_val = col.mean()
            print(f"    {key:30s}: mean={mean_val:.3f}  NaN={nan_cnt}/{len(col)}")
        else:
            print(f"    {key:30s}: 컬럼 없음")

    def _safe_score(key: str) -> float:
        """NaN 개수 로그 출력 후 유효 샘플 평균 반환 (전체 NaN 이면 0.0)"""
        try:
            if key in result_df_raw.columns:
                valid = result_df_raw[key].dropna()
                return float(valid.mean()) if len(valid) > 0 else 0.0
            return float(result[key])
        except Exception:
            return 0.0

    scores = {
        "context_precision": _safe_score("context_precision"),
        "context_recall"   : _safe_score("context_recall"),
        "faithfulness"     : _safe_score("faithfulness"),
        "answer_relevancy" : _safe_score("answer_relevancy"),
    }
    print_score_summary(scores, label=f"RAG 평가 결과 ({label})")

    # STEP 7: 저장
    result_df = result.to_pandas()
    result_df["eval_strategy"] = label
    save_results(result_df, RAG_EVAL_PATH, "RAG 평가 결과")

    print("\n" + "=" * 65)
    print("완료!")
    print("=" * 65)


if __name__ == "__main__":
    main()