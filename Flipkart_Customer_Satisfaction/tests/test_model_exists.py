
import joblib, pathlib
def test_model_file_exists():
    assert pathlib.Path(__file__).resolve().parents[1].joinpath('models','rf_model.joblib').exists()
