"""
test/test_loader.py
src/processing/Loader.py 의 파일명 파싱 로직(parse_filename)에 대한 단위 테스트.
파일시스템/네트워크/LLM 호출 없이 순수하게 문자열 파싱만 검증한다.
"""
from src.processing.Loader import parse_filename


class TestParseFilename:
    def test_parses_date_sector_title(self):
        meta = parse_filename(
            "260414_DS투자증권_반도체_낸(NAND)붐온.pdf",
            folder_name="DS투자증권",
        )
        assert meta["source_firm"] == "DS투자증권"
        assert meta["report_date"] == "2026-04-14"
        assert meta["sector"] == "반도체"
        assert "낸(NAND)붐온" in meta["title"]

    def test_uses_folder_name_as_source_firm(self):
        # source_firm 은 파일명이 아니라 폴더명을 기준으로 함
        meta = parse_filename("260414_아무증권_기타_제목.pdf", folder_name="키움증권")
        assert meta["source_firm"] == "키움증권"

    def test_invalid_date_prefix_returns_none_date(self):
        # 날짜 형식(6자리 숫자)이 아니면 report_date 는 None 이어야 함
        meta = parse_filename("이상한파일명.pdf", folder_name="키움증권")
        assert meta["report_date"] is None

    def test_missing_parts_do_not_crash(self):
        # 언더스코어가 부족한 파일명이어도 예외 없이 처리되어야 함
        meta = parse_filename("260414.pdf", folder_name="키움증권")
        assert meta["report_date"] == "2026-04-14"
        assert meta["sector"] is None
        assert meta["title"] is None
