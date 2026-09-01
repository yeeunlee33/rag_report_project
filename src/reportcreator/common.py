"""
src/reportcreator/common.py
공통 유틸: 섹터/증권사 정규화, LLM 인스턴스 헬퍼, 문서 메타데이터 헬퍼

freeform_chain.py 리팩터링(2,181줄 → 역할별 분리)으로 만들어진 파일입니다.
로직은 원본에서 그대로 옮겨온 것이며 변경되지 않았습니다.
"""
from __future__ import annotations

from langchain.schema import Document
from langchain_openai import ChatOpenAI


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

