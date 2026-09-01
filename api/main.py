"""
api/main.py
RAG 파이프라인 FastAPI 서버

엔드포인트:
  GET  /health    → DB 연결 상태 헬스체크
  GET  /info      → 시스템 정보 (전략 설정, 청크 수 등)
  POST /search    → 하이브리드 검색 (청크 레벨)
  POST /report    → 분석 리포트 생성 (freeform_chain 7종)

운영 기본세트:
  - 요청별 request_id 자동 부여 → 로그/트레이싱 추적
  - 구조화 로깅 (JSON-like, timestamp · method · path · status · latency)
  - LLM 호출 타임아웃 (기본 120s, REPORT_TIMEOUT_SEC 환경변수로 조절)
  - 지수 백오프 재시도 (tenacity — OpenAI rate limit / 일시적 네트워크 오류 대응)
  - 글로벌 예외 핸들러 → 500도 JSON 포맷 반환 (스택 트레이스 숨김)
  - 입력 검증 (Pydantic) + 빈 쿼리 / 범위 초과 차단

실행:
  uvicorn api.main:app --reload --port 8000

Swagger UI:
  http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag.api")

# ── 환경변수 설정 ─────────────────────────────────────────────────────────────
REPORT_TIMEOUT_SEC = int(os.getenv("REPORT_TIMEOUT_SEC", "120"))  # LLM 호출 타임아웃
SEARCH_TIMEOUT_SEC = int(os.getenv("SEARCH_TIMEOUT_SEC", "30"))   # 검색 타임아웃
MAX_QUESTION_LEN   = int(os.getenv("MAX_QUESTION_LEN",   "500"))  # 최대 질문 길이


# ── 글로벌 상태 (startup에서 초기화) ─────────────────────────────────────────
_state: dict[str, Any] = {
    "retrievers":    None,
    "reranker":      None,
    "total_chunks":  0,
    "db_path":       "",
    "chunking":      "",
    "embedding":     "",
    "vectorstore":   "",
    "ready":         False,
}


# ── 재시도 데코레이터 (OpenAI rate limit · 일시적 오류 대응) ──────────────────
# wait_exponential: 1s → 2s → 4s → ... 최대 10s 대기, 최대 3번 재시도
_RETRYABLE = (Exception,)  # 필요 시 openai.RateLimitError 등으로 좁힐 수 있음

def _is_retryable(exc: Exception) -> bool:
    """재시도할 예외인지 판별. 4xx 에러(입력 오류)는 재시도하지 않음."""
    if isinstance(exc, HTTPException):
        return False
    msg = str(exc).lower()
    retryable_keywords = ("rate limit", "timeout", "connection", "service unavailable", "503", "429")
    return any(kw in msg for kw in retryable_keywords)


def with_retry(fn):
    """LLM 호출 함수에 재시도 로직을 적용하는 데코레이터."""
    return retry(
        retry=retry_if_exception_type(Exception) if False else retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )(fn)


# ── 파이프라인 초기화 ─────────────────────────────────────────────────────────

def _load_pipeline() -> bool:
    """
    startup 시 한 번만 실행. 벡터스토어 로드 + 리트리버 인덱스 구성.
    실패하면 False 반환 → /health에서 503 반환.
    """
    try:
        from langchain.schema import Document

        from src.processing.chunking import chunking_03_hybrid    as CHUNKING
        from src.embedding            import embedding_01_openai   as EMBEDDING
        from src.vectorstore          import vectorstore_01_chroma as VECTORSTORE
        from src.retriever            import router                as ROUTER
        from src.reranker             import reranker_01_crossencoder as RERANKER

        VS_BASE_DIR = PROJECT_ROOT / "data" / "vectorstore"
        db_path = str(
            VS_BASE_DIR / VECTORSTORE.STRATEGY_NAME
                        / EMBEDDING.STRATEGY_NAME
                        / CHUNKING.STRATEGY_NAME
        )

        if not VECTORSTORE.exists(db_path):
            logger.warning("ChromaDB 없음: %s — /health 503 반환", db_path)
            _state["ready"] = False
            return False

        logger.info("임베딩 로드 중... (%s)", EMBEDDING.STRATEGY_NAME)
        embeddings  = EMBEDDING.get_embeddings()

        logger.info("벡터스토어 로드 중... (%s)", VECTORSTORE.STRATEGY_NAME)
        vectorstore = VECTORSTORE.load(db_path, embeddings)

        logger.info("BM25 인덱스 구성 중...")
        results  = vectorstore.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"], results["metadatas"])
        ]
        retrievers = ROUTER.build_retriever(vectorstore, all_docs, k=40)

        _state.update(
            retrievers   = retrievers,
            reranker     = RERANKER,
            total_chunks = len(all_docs),
            db_path      = db_path,
            chunking     = CHUNKING.STRATEGY_NAME,
            embedding    = EMBEDDING.STRATEGY_NAME,
            vectorstore  = VECTORSTORE.STRATEGY_NAME,
            ready        = True,
        )
        logger.info("파이프라인 초기화 완료 — 총 %d개 청크", len(all_docs))
        return True

    except Exception:
        logger.error("파이프라인 초기화 실패:\n%s", traceback.format_exc())
        _state["ready"] = False
        return False


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 파이프라인 초기화 (블로킹 → 별도 스레드에서 실행)
    logger.info("서버 시작 — 파이프라인 로드 중...")
    await asyncio.get_event_loop().run_in_executor(None, _load_pipeline)
    yield
    # shutdown: 필요 시 정리 작업
    logger.info("서버 종료")


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ResearchRAG API",
    description = "증권사 리서치 리포트 RAG 시스템 — 검색 및 리포트 생성",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS (스테이징 환경에서 프론트엔드 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── 미들웨어: 요청 로깅 + request_id 주입 ─────────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    모든 요청에 대해:
    1. X-Request-ID 헤더 부여 (없으면 자동 생성)
    2. 요청 시작/종료 로그 기록
    3. 응답에 X-Request-ID 헤더 추가 (클라이언트가 추적 가능)
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id

    start = time.perf_counter()
    logger.info(
        "→ %s %s | req_id=%s | client=%s",
        request.method, request.url.path, request_id,
        request.client.host if request.client else "unknown",
    )

    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "← %s %s | req_id=%s | status=%d | %.1fms",
        request.method, request.url.path, request_id,
        response.status_code, latency_ms,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
    return response


# ── 글로벌 예외 핸들러 ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    처리되지 않은 예외를 JSON으로 반환.
    스택 트레이스는 로그에만 기록하고 응답에는 포함하지 않음 (보안).
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled error | req_id=%s\n%s", request_id, traceback.format_exc())
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {
            "error":      "internal_server_error",
            "message":    "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "request_id": request_id,
        },
    )


# ── 헬퍼: 파이프라인 준비 확인 ───────────────────────────────────────────────

def _require_pipeline():
    """파이프라인이 준비되지 않았으면 503 반환."""
    if not _state["ready"]:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "파이프라인이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
        )


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length  = 1,
        max_length  = MAX_QUESTION_LEN,
        description = "검색 쿼리",
        examples    = ["반도체 업황 전망"],
    )
    top_n: int = Field(
        default     = 10,
        ge          = 1,
        le          = 30,
        description = "반환할 청크 수 (1~30)",
    )


class SearchResultItem(BaseModel):
    rank:         int
    source_firm:  str
    report_date:  str
    sector:       str
    title:        str
    content:      str
    rerank_score: float | None


class SearchResponse(BaseModel):
    request_id:   str
    query:        str
    question_type: str
    target_sector: str | None
    target_period: str | None
    target_brokers: list[str]
    total:        int
    results:      list[SearchResultItem]
    latency_ms:   float


class ReportRequest(BaseModel):
    question: str = Field(
        ...,
        min_length  = 2,
        max_length  = MAX_QUESTION_LEN,
        description = "분석 질문",
        examples    = ["하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘"],
    )


class ReportResponse(BaseModel):
    request_id:    str
    question:      str
    question_type: str
    sources:       list[str]
    answer:        str
    latency_ms:    float


class HealthResponse(BaseModel):
    status:       str          # "ok" | "degraded"
    ready:        bool
    total_chunks: int
    db_path:      str


class InfoResponse(BaseModel):
    chunking:    str
    embedding:   str
    vectorstore: str
    reranker:    str
    total_chunks: int
    timeout: dict[str, int]


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "헬스체크",
    description    = "DB 연결 상태와 파이프라인 준비 여부를 반환합니다.",
    tags           = ["운영"],
)
async def health():
    return HealthResponse(
        status       = "ok" if _state["ready"] else "degraded",
        ready        = _state["ready"],
        total_chunks = _state["total_chunks"],
        db_path      = _state["db_path"],
    )


@app.get(
    "/info",
    response_model = InfoResponse,
    summary        = "시스템 정보",
    description    = "현재 적용된 전략 설정과 청크 수를 반환합니다.",
    tags           = ["운영"],
)
async def info():
    reranker_name = (
        _state["reranker"].STRATEGY_NAME
        if _state["reranker"] and hasattr(_state["reranker"], "STRATEGY_NAME")
        else "unknown"
    )
    return InfoResponse(
        chunking     = _state["chunking"],
        embedding    = _state["embedding"],
        vectorstore  = _state["vectorstore"],
        reranker     = reranker_name,
        total_chunks = _state["total_chunks"],
        timeout      = {
            "report_sec": REPORT_TIMEOUT_SEC,
            "search_sec": SEARCH_TIMEOUT_SEC,
        },
    )


@app.post(
    "/search",
    response_model = SearchResponse,
    summary        = "하이브리드 검색",
    description    = (
        "쿼리를 분석해 BM25 + 벡터 앙상블 검색을 실행하고 "
        "Cross-Encoder로 리랭킹한 청크 목록을 반환합니다."
    ),
    tags           = ["검색"],
)
async def search(req: SearchRequest, request: Request):
    _require_pipeline()
    request_id = getattr(request.state, "request_id", "unknown")

    # ── 실제 검색 로직 (동기 함수 → run_in_executor로 비동기 처리) ──────────
    def _run_search():
        from src.reportcreator.freeform_chain import _analyze_intent, _collect_chunks

        intent_data = _analyze_intent(req.query)
        retriever_intent = (
            "balanced"
            if intent_data["question_type"] in ("broker_comparison", "consensus", "other")
            else "ensemble"
        )

        docs = _collect_chunks(
            _state["retrievers"],
            queries        = intent_data["search_queries"],
            target_brokers = intent_data["target_brokers"],
            target_period  = intent_data.get("target_period"),
            target_sector  = intent_data.get("target_sector"),
            rerank_fn      = _state["reranker"].rerank,
            k_per_query    = 40,
            top_n          = req.top_n,
            intent         = retriever_intent,
        )
        return docs, intent_data

    t0 = time.perf_counter()
    logger.info("검색 시작 | req_id=%s | query=%r | top_n=%d", request_id, req.query, req.top_n)

    try:
        # SEARCH_TIMEOUT_SEC 초 안에 끝나지 않으면 TimeoutError
        docs, intent_data = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _run_search),
            timeout = SEARCH_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("검색 타임아웃 | req_id=%s | %ds 초과", request_id, SEARCH_TIMEOUT_SEC)
        raise HTTPException(
            status_code = status.HTTP_504_GATEWAY_TIMEOUT,
            detail      = f"검색이 {SEARCH_TIMEOUT_SEC}초를 초과했습니다. 쿼리를 단순화해보세요.",
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "검색 완료 | req_id=%s | results=%d | %.0fms",
        request_id, len(docs), latency_ms,
    )

    results = [
        SearchResultItem(
            rank         = i + 1,
            source_firm  = doc.metadata.get("source_firm", "-"),
            report_date  = doc.metadata.get("report_date", "-"),
            sector       = doc.metadata.get("sector", "-"),
            title        = doc.metadata.get("title", ""),
            content      = doc.page_content[:500],  # 미리보기 500자
            rerank_score = (
                float(doc.metadata["rerank_score"])
                if "rerank_score" in doc.metadata else None
            ),
        )
        for i, doc in enumerate(docs)
    ]

    return SearchResponse(
        request_id     = request_id,
        query          = req.query,
        question_type  = intent_data["question_type"],
        target_sector  = intent_data.get("target_sector"),
        target_period  = intent_data.get("target_period"),
        target_brokers = intent_data["target_brokers"],
        total          = len(results),
        results        = results,
        latency_ms     = round(latency_ms, 1),
    )


@app.post(
    "/report",
    response_model = ReportResponse,
    summary        = "분석 리포트 생성",
    description    = (
        "질문을 7종 유형(fact_lookup · coverage_summary · timeline · "
        "broker_comparison · risk · consensus · other)으로 분류한 뒤 "
        "증권사 리포트 발췌를 바탕으로 분석 리포트를 생성합니다."
    ),
    tags           = ["리포트"],
)
async def report(req: ReportRequest, request: Request):
    _require_pipeline()
    request_id = getattr(request.state, "request_id", "unknown")

    # ── 재시도 포함 LLM 호출 래퍼 ─────────────────────────────────────────
    @with_retry
    def _run_report():
        from src.reportcreator.freeform_chain import answer_question
        return answer_question(
            _state["retrievers"],
            req.question,
            rerank_fn = _state["reranker"].rerank,
        )

    t0 = time.perf_counter()
    logger.info(
        "리포트 생성 시작 | req_id=%s | question=%r",
        request_id, req.question[:80],
    )

    try:
        # REPORT_TIMEOUT_SEC 초 안에 끝나지 않으면 TimeoutError
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _run_report),
            timeout = REPORT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "리포트 타임아웃 | req_id=%s | %ds 초과",
            request_id, REPORT_TIMEOUT_SEC,
        )
        raise HTTPException(
            status_code = status.HTTP_504_GATEWAY_TIMEOUT,
            detail      = (
                f"리포트 생성이 {REPORT_TIMEOUT_SEC}초를 초과했습니다. "
                "질문을 더 구체적으로 작성해보세요."
            ),
        )
    except Exception as exc:
        logger.error("리포트 생성 실패 | req_id=%s | %s", request_id, exc)
        raise HTTPException(
            status_code = status.HTTP_502_BAD_GATEWAY,
            detail      = f"LLM 호출 중 오류가 발생했습니다: {type(exc).__name__}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "리포트 생성 완료 | req_id=%s | type=%s | sources=%s | %.0fms",
        request_id,
        result.get("question_type", "?"),
        result.get("sources", []),
        latency_ms,
    )

    return ReportResponse(
        request_id    = request_id,
        question      = req.question,
        question_type = result.get("question_type", "other"),
        sources       = result.get("sources", []),
        answer        = result.get("answer", ""),
        latency_ms    = round(latency_ms, 1),
    )
