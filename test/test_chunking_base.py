"""
test/test_chunking_base.py
src/processing/chunking/base.py 의 순수 유틸 함수에 대한 단위 테스트.
"""
from src.processing.chunking.base import make_chunk_id, extract_meta


class TestMakeChunkId:
    def test_basic_id(self):
        assert make_chunk_id("하나증권", "2026-04-14", 0) == "하나증권_20260414_0"

    def test_removes_spaces_in_firm_name(self):
        assert make_chunk_id("DS 투자증권", "2026-04-14", 1) == "DS투자증권_20260414_1"

    def test_missing_date_uses_nodate(self):
        assert make_chunk_id("키움증권", None, 2) == "키움증권_nodate_2"

    def test_prefix_is_included(self):
        assert make_chunk_id("키움증권", "2026-04-14", 0, prefix="parent") == "키움증권_20260414_parent_0"


class TestExtractMeta:
    def test_extracts_known_keys_only(self):
        report = {
            "source_firm": "하나증권",
            "report_date": "2026-04-14",
            "sector": "반도체",
            "title": "제목",
            "extra_unrelated_key": "포함되면 안 됨",
        }
        meta = extract_meta(report)
        assert meta["source_firm"] == "하나증권"
        assert meta["sector"] == "반도체"
        assert "extra_unrelated_key" not in meta

    def test_missing_keys_default_safely(self):
        meta = extract_meta({})
        assert meta["source_firm"] == ""
        assert meta["report_date"] is None
        assert meta["filename"] == ""
