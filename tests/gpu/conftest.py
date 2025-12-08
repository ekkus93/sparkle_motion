from __future__ import annotations

import pytest

from . import helpers

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _skip_without_gpu() -> None:
    helpers.require_gpu_available()
