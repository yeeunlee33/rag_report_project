"""
test/test_intent.py
src/reportcreator/intent.py 의 순수 함수(_normalize_label, _target_tokens)에 대한 단위 테스트.
LLM 호출(_analyze_intent)은 대상에서 제외한다 — 실제 OpenAI API 호출이 필요해서
비용이 들고, CI에 API 키를 노출해야 하는 문제가 생기기 때문.
"""
from src.reportcreator.intent import _normalize_label, _target_tokens


class TestNormalizeLabel:
    def test_removes_whitespace_and_증권(self):
        assert _normalize_label("하나 증권") == "하나"
        assert _normalize_label("키움증권") == "키움"

    def test_lowercases(self):
        assert _normalize_label("Hana") == "hana"

    def test_non_string_returns_empty(self):
        assert _normalize_label(None) == ""
        assert _normalize_label(123) == ""


class TestTargetTokens:
    def test_splits_compound_expression(self):
        tokens = _target_tokens("반도체/디스플레이")
        assert "반도체" in tokens
        assert "디스플레이" in tokens

    def test_removes_stopwords(self):
        tokens = _target_tokens("반도체 섹터 전망")
        assert "반도체" in tokens
        # "섹터", "전망" 은 stopwords 이므로 토큰으로 남으면 안 됨
        assert all(t not in ("섹터", "전망") for t in tokens)

    def test_full_normalized_value_included_first(self):
        tokens = _target_tokens("HBM")
        assert tokens[0] == _normalize_label("HBM")

    def test_empty_input_returns_empty_list(self):
        assert _target_tokens("") == []
        assert _target_tokens(None) == []
        assert _target_tokens("   ") == []
