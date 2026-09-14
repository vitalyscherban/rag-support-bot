import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import pytest  # noqa: E402

from supportbot.corpus import load_corpus  # noqa: E402
from supportbot.embeddings import HashingEmbeddings  # noqa: E402
from supportbot.pipeline import SupportBot  # noqa: E402


@pytest.fixture(scope="session")
def documents() -> dict[str, str]:
    return load_corpus()


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
def bot(documents) -> SupportBot:
    return SupportBot(documents)
