import pytest


@pytest.fixture
def user_data() -> dict[str, object]:
    return {"name": "Tom", "age": 18}