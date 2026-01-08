"""
pytest configuration and fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root_dir):
    """Return data directory path."""
    return project_root_dir / "data"


@pytest.fixture(scope="session")
def index_dir(data_dir):
    """Return indices directory path."""
    return data_dir / "indices"
