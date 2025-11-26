"""Local ADK shim package for testing.

This lightweight shim exists only to allow local/dev tests to exercise
the SDK-path code paths without requiring real `google.adk` installation
or credentials. It intentionally provides minimal, safe implementations
that write artifacts to the local filesystem.
"""

__all__ = []
