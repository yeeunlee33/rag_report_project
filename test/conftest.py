"""
test/conftest.py
pytest 전역 설정.

src/processing/Loader.py 는 모듈을 import 하는 시점에
`client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))` 를 실행한다.
로컬에는 .env 에 실제 키가 있어 문제없지만, CI 환경에는 실제 키가 없으므로
이 값이 없으면 import 자체가 예외를 던져서 테스트 수집(collection)이 실패한다.

여기서는 실제 OpenAI API를 호출하지 않는 순수 함수(parse_filename 등)만
테스트하므로, 키가 없을 때는 더미 값으로 채워 import 실패를 막는다.
(.env 에 실제 키가 있으면 그 값이 우선 사용된다 — setdefault 이므로 덮어쓰지 않음)
"""
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-ci")
