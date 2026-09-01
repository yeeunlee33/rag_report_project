"""
test/test_cleaner.py
src/processing/cleaner.py 의 정규식 기반 정제 함수에 대한 단위 테스트.
"""
from src.processing.cleaner import _remove_page_numbers, _strip_markdown_syntax


class TestRemovePageNumbers:
    def test_removes_dash_wrapped_number(self):
        text = "본문 내용\n- 3 -\n다음 내용"
        result = _remove_page_numbers(text)
        assert "- 3 -" not in result

    def test_removes_page_of_total_format(self):
        text = "본문 내용\nPage 2 of 10\n다음 내용"
        result = _remove_page_numbers(text)
        assert "Page 2 of 10" not in result

    def test_keeps_normal_sentences_untouched(self):
        text = "이 리포트는 2026년 반도체 업황을 다룬다."
        assert _remove_page_numbers(text) == text


class TestStripMarkdownSyntax:
    def test_removes_heading_marks(self):
        assert _strip_markdown_syntax("## 핵심 요약") == "핵심 요약"

    def test_removes_bold_and_italic(self):
        assert _strip_markdown_syntax("**중요**한 내용과 *강조*") == "중요한 내용과 강조"

    def test_removes_inline_code_backticks(self):
        assert _strip_markdown_syntax("`코드` 표기") == "코드 표기"

    def test_removes_horizontal_rule(self):
        text = "내용1\n---\n내용2"
        result = _strip_markdown_syntax(text)
        assert "---" not in result
