from model_utils import update_model
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    format = '%(asctime)s %(levelname)s: %(name)s: %(message)s',
    level = logging.INFO,
    datefmt = '%H:%M:%S',
    stream = sys.stderr
)

logging.getLogger(__name__)
logging.info("Loading Data...")

data = pd.read_csv("dataset/full_data.csv")

logging.info("Loading Model...")

model = Pipeline([
    ("imputer", SimpleImputer(strategy = "mean", missing_values = np.nan)),
    ("core_model", GradientBoostingRegressor()),
])

logging.info("Splitting Dataset...")

X = data.drop(["worldwide_gross"], axis = 1)
y = data["worldwide_gross"]

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)

logging.info("Setting Hyperparameters...")

param_tuning = {"core_model__n_estimators" : range(20, 301, 20)}

grid_search = GridSearchCV(model, param_grid = param_tuning, scoring = "r2", cv = 5)

logging.info("Starting Grid Search...")

grid_search.fit(X_train, y_train)

logging.info("Cross Validating Best Model...")

final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score = True, cv = 5)

train_score = np.mean(final_result["train_score"])
test_score = np.mean(final_result["test_score"])
assert train_score > 0.7
assert test_score > 0.65

logging.info(f"Train Score: {train_score}")
logging.info(f"Test Score: {test_score}")

logging.info("Updating Model...")

update_model(grid_search.best_estimator_)

breakpoint()