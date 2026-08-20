# Copyright (c) 2026 BAAI. All rights reserved.

"""
Global pytest fixtures and configuration for all tests.

This root-level conftest is loaded by pytest before any sub-package conftest.
It provides:
  - Custom markers auto-registration
"""


def pytest_configure(config):
    """Register all custom markers in a single place."""
    for line in [
        "benchmark: marks tests as benchmark smoke tests",
        "e2e: marks tests as end-to-end smoke tests",
        "functional: marks tests as functional correctness tests",
        "gpu: marks tests that require a GPU",
    ]:
        config.addinivalue_line("markers", line)
