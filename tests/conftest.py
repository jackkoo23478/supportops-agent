import pytest

from supportops.store import Store


@pytest.fixture
def store() -> Store:
    """Store backed by the sample data in data/sample."""
    return Store()
