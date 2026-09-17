from collections.abc import Iterator
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = Path.cwd() / ".test-work" / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        rmtree(path)
