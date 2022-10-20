from sklearn.pipeline import Pipeline
from joblib import dump

def update_model(model: Pipeline) -> None:
    dump(model, "model/model.pkl")