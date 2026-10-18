import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from termmap.util import load_config  # noqa: E402

HAVE_DATA = (ROOT / "data" / "hackathon_train.csv").exists() and \
    (ROOT / "resources" / "icd10cm_codes.xlsx").exists() and \
    (ROOT / "resources" / "SNOMEDCT_US_codes_and_descriptions.xlsx").exists()
HAVE_MODEL = (ROOT / "models" / "final.pkl").exists()

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="challenge files not present (never committed)")
needs_model = pytest.mark.skipif(not (HAVE_DATA and HAVE_MODEL), reason="run `make train` first")


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/evaluation.yaml")


@pytest.fixture(scope="session")
def R(cfg):
    if not HAVE_DATA:
        pytest.skip("no data")
    from termmap.pipeline import load_resources

    return load_resources(cfg)


@pytest.fixture(scope="session")
def model():
    if not HAVE_MODEL:
        pytest.skip("no model")
    from termmap.pipeline import load_model

    return load_model(ROOT / "models" / "final.pkl")


@pytest.fixture(scope="session")
def predictor(R, model):
    from termmap.pipeline import Predictor

    return Predictor(R, model)
