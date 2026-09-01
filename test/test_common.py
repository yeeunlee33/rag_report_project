"""
test/test_common.py
src/reportcreator/common.py 에 대한 단위 테스트.
LLM 호출 없이 순수 로직만 검증한다 (API 키 불필요, CI에서 바로 실행 가능).
"""
from langchain.schema import Document

from src.reportcreator.common import normalize_sector, _get_firm


class TestNormalizeSector:
    def test_alias_is_mapped(self):
        # HBM 은 SECTOR_ALIASES 에 정의된 별칭 -> 반도체로 정규화되어야 함
        assert normalize_sector("HBM") == "반도체"
        assert normalize_sector("완성차") == "자동차"
        assert normalize_sector("조선업") == "조선"

    def test_valid_sector_passes_through(self):
        # VALID_SECTORS 에 이미 있는 값은 그대로 반환
        assert normalize_sector("반도체") == "반도체"
        assert normalize_sector("자동차") == "자동차"

    def test_unknown_sector_falls_back_to_other(self):
        # 별칭도 아니고 유효 섹터도 아니면 "기타"로 폴백
        assert normalize_sector("존재하지않는섹터") == "기타"

    def test_empty_or_none_returns_none(self):
        assert normalize_sector(None) is None
        assert normalize_sector("") is None


class TestGetFirm:
    def test_prefers_source_firm(self):
        doc = Document(page_content="본문", metadata={"source_firm": "하나증권", "broker": "키움증권"})
        assert _get_firm(doc) == "하나증권"

    def test_falls_back_to_broker(self):
        doc = Document(page_content="본문", metadata={"broker": "키움증권"})
        assert _get_firm(doc) == "키움증권"

    def test_strips_whitespace(self):
        doc = Document(page_content="본문", metadata={"source_firm": "  하나증권  "})
        assert _get_firm(doc) == "하나증권"

    def test_falsy_or_missing_values_return_unknown(self):
        # 빈 문자열/0/False 같은 값이 metadata 에 들어있어도 "알 수 없음" 으로 방어되어야 함
        doc = Document(page_content="본문", metadata={"source_firm": "", "broker": None})
        assert _get_firm(doc) == "알 수 없음"

        doc_empty = Document(page_content="본문", metadata={})
        assert _get_firm(doc_empty) == "알 수 없음"
