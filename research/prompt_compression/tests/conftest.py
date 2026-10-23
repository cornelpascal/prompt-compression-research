from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def pc_models(tmp_path):
    from promptcomp.smoke import tiny_models
    return tiny_models(tmp_path / "models")


@pytest.fixture
def pc_data(tmp_path):
    from promptcomp.smoke import fixture_data
    return fixture_data(tmp_path / "data")

