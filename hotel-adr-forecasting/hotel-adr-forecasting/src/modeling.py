"""Model definitions, evaluation metrics and hyper-parameter tuning."""
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUM = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
            "is_repeated_guest", "total_of_special_requests", "required_car_parking_spaces",
            "arrival_date_week_number", "is_canceled"]
BASE_CAT = ["hotel", "arrival_date_month", "market_segment", "reserved_room_type",
            "deposit_type", "customer_type"]
ENG_NUM = ["total_nights", "total_guests", "month_sin", "month_cos", "has_weekend",
           "log_lead_time", "has_children", "peak_season"]


def preprocessor(num_cols, cat_cols, scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])


def baseline_models():
    """Stage 1: baseline models on raw features only."""
    return {
        "Mean predictor": Pipeline([("p", preprocessor(BASE_NUM, BASE_CAT)), ("m", DummyRegressor())]),
        "Linear Regression": Pipeline([("p", preprocessor(BASE_NUM, BASE_CAT, True)), ("m", LinearRegression())]),
        "Random Forest (default)": Pipeline([("p", preprocessor(BASE_NUM, BASE_CAT)),
                                             ("m", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))]),
    }


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4), "MAPE_%": round(mape, 2)}


def tune_gradient_boosting(X, y, n_iter=25, seed=42):
    """Stage 3: RandomizedSearchCV over HistGradientBoosting using engineered features."""
    pipe = Pipeline([("p", preprocessor(BASE_NUM + ENG_NUM, BASE_CAT)),
                     ("m", HistGradientBoostingRegressor(random_state=seed, early_stopping=True))])
    space = {
        "m__learning_rate": loguniform(0.02, 0.3),
        "m__max_iter": randint(200, 800),
        "m__max_leaf_nodes": randint(15, 80),
        "m__min_samples_leaf": randint(10, 80),
        "m__l2_regularization": uniform(0, 2),
    }
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=KFold(5, shuffle=True, random_state=seed),
                                scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=seed, verbose=1)
    return search.fit(X, y)


def tune_random_forest(X, y, n_iter=10, seed=42):
    """Tuned Random Forest for a like-for-like comparison."""
    pipe = Pipeline([("p", preprocessor(BASE_NUM + ENG_NUM, BASE_CAT)),
                     ("m", RandomForestRegressor(n_jobs=-1, random_state=seed))])
    space = {"m__n_estimators": randint(150, 400), "m__max_depth": randint(10, 30),
             "m__min_samples_leaf": randint(1, 8), "m__max_features": uniform(0.3, 0.6)}
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=KFold(3, shuffle=True, random_state=seed),
                                scoring="neg_root_mean_squared_error", n_jobs=1, random_state=seed)
    return search.fit(X, y)
