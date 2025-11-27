from __future__ import annotations

import os


def test_model_context_calls_close(monkeypatch):
    called = {"closed": False}

    class FakeModel:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            called["closed"] = True

    def loader():
        return FakeModel()

    from sparkle_motion.gpu_utils import model_context

    with model_context(loader) as m:
        assert hasattr(m, "close")

    assert called["closed"] is True


def test_model_context_fixture_mode(monkeypatch):
    os.environ["ADK_USE_FIXTURE"] = "1"

    class FakeModel:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def loader():
        return FakeModel()

    from sparkle_motion.gpu_utils import model_context

    with model_context(loader) as m:
        assert hasattr(m, "close")
    # In fixture mode teardown is a no-op (we don't require close to be called)
    del os.environ["ADK_USE_FIXTURE"]
