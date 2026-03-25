"""
Pytest configuration.
Sets asyncio mode to 'auto' so all async tests run without extra decorators.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
