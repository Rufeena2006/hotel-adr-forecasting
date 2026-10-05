"""End-to-end pipeline: baseline -> evaluation -> tuning -> business analysis.

Usage:  python run_pipeline.py [--quick]
"""
import argparse, json, time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data import load_raw, clean, engineer, TARGET, MONTHS
from src.modeling import (baseline_models, regression_metrics, tune_gradient_boosting,
                          tune_random_forest, BASE_NUM, BASE_CAT, ENG_NUM)

OUT = Path("outputs"); FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def main(quick: bool):
    t0 = time.time()
    raw, source = load_raw()
    df = engineer(clean(raw))
    print(f"Data source: {source} | rows after cleaning: {len(df):,}")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results, preds = {}, {}
    # ---- Stage 1: baselines ------------------------------------------------
    for name, model in baseline_models().items():
        model.fit(X_tr, y_tr)
        preds[name] = model.predict(X_te)
        results[name] = regression_metrics(y_te.values, preds[name])
        print(f"{name:28s}", results[name])

    # ---- Stage 2/3: feature engineering + hyper-parameter tuning -------------
    gb = tune_gradient_boosting(X_tr, y_tr, n_iter=8 if quick else 25)
    best = gb.best_estimator_
    preds["Tuned Gradient Boosting"] = best.predict(X_te)
    results["Tuned Gradient Boosting"] = regression_metrics(y_te.values, preds["Tuned Gradient Boosting"])
    print("Best params:", gb.best_params_, "| CV RMSE:", round(-gb.best_score_, 3))
    print(f"{'Tuned Gradient Boosting':28s}", results["Tuned Gradient Boosting"])

    rf = tune_random_forest(X_tr, y_tr, n_iter=3 if quick else 8)
    preds["Tuned Random Forest"] = rf.best_estimator_.predict(X_te)
    results["Tuned Random Forest"] = regression_metrics(y_te.values, preds["Tuned Random Forest"])
    print(f"{'Tuned Random Forest':28s}", results["Tuned Random Forest"])

    # ---- Champion model ----------------------------------------------------
    champion = min((k for k in results if k != "Mean predictor"), key=lambda k: results[k]["RMSE"])
    champ_pred = preds[champion]
    print("Champion:", champion)
    joblib.dump(best, OUT / "best_model.joblib")

    # ---- Permutation importance ---------------------------------------------
    cols = BASE_NUM + ENG_NUM + BASE_CAT
    imp = permutation_importance(best, X_te[cols], y_te, n_repeats=5, random_state=42, n_jobs=-1,
                                 scoring="neg_root_mean_squared_error")
    imp_s = pd.Series(imp.importances_mean, index=cols).sort_values()

    # ---- Segment error analysis -------------------------------------------
    te = X_te.copy(); te["actual"] = y_te.values; te["pred"] = preds["Tuned Gradient Boosting"]
    te["abs_err"] = (te.actual - te.pred).abs()
    seg_err = te.groupby("market_segment").agg(MAE=("abs_err", "mean"), mean_adr=("actual", "mean"),
                                               n=("actual", "size")).round(2)
    hotel_err = te.groupby("hotel").agg(MAE=("abs_err", "mean"), mean_adr=("actual", "mean")).round(2)

    # ---- Business view: seasonal ADR + revenue exposure ----------------------
    monthly = (te.groupby("arrival_date_month")[["actual", "pred"]].mean().reindex(MONTHS).round(2))
    nights = te["total_nights"]
    revenue_actual = float((te.actual * nights).sum())
    base_err = float(((y_te.values - preds["Linear Regression"]).__abs__() * nights).sum())
    tuned_err = float((te.abs_err * nights).sum())
    business = {
        "test_revenue_actual": round(revenue_actual, 0),
        "pricing_error_exposure_baseline_LR": round(base_err, 0),
        "pricing_error_exposure_tuned": round(tuned_err, 0),
        "exposure_reduction_pct": round((1 - tuned_err / base_err) * 100, 1),
        "baseline_exposure_pct_of_revenue": round(base_err / revenue_actual * 100, 2),
        "tuned_exposure_pct_of_revenue": round(tuned_err / revenue_actual * 100, 2),
    }

    # ---- Figures -----------------------------------------------------------
    names = list(results); rmse = [results[n]["RMSE"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, rmse, color=["#9aa5b1"] * (len(names) - 2) + ["#2a7de1", "#16a085"])
    for i, v in enumerate(rmse): ax.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("RMSE (EUR/GBP per night, lower is better)"); ax.set_title("Model comparison on hold-out set")
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png"); plt.close(fig)

    p = preds["Tuned Gradient Boosting"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, p, s=4, alpha=0.3, color="#2a7de1")
    lim = [0, max(y_te.max(), p.max())]; ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Actual ADR"); ax.set_ylabel("Predicted ADR"); ax.set_title("Actual vs predicted (tuned GB)")
    fig.tight_layout(); fig.savefig(FIG / "actual_vs_predicted.png"); plt.close(fig)

    res = y_te.values - p
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].hist(res, bins=50, color="#16a085"); axs[0].set_title("Residual distribution"); axs[0].set_xlabel("Residual")
    axs[1].scatter(p, res, s=4, alpha=0.3, color="#16a085"); axs[1].axhline(0, color="r", lw=1)
    axs[1].set_title("Residuals vs predicted"); axs[1].set_xlabel("Predicted ADR")
    fig.tight_layout(); fig.savefig(FIG / "residuals.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5)); imp_s.tail(12).plot.barh(ax=ax, color="#8e44ad")
    ax.set_title("Permutation importance (RMSE increase)"); fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(MONTHS, monthly.actual, marker="o", label="Actual"); ax.plot(MONTHS, monthly.pred, marker="s", ls="--", label="Predicted")
    ax.set_xticks(range(12)); ax.set_xticklabels([m[:3] for m in MONTHS]); ax.set_ylabel("Mean ADR")
    ax.set_title("Seasonal ADR: actual vs model"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "seasonal_adr.png"); plt.close(fig)

    # ---- Persist ----------------------------------------------------------
    summary = {"data_source": source, "n_rows": int(len(df)), "results": results, "champion": champion,
               "best_params": {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in gb.best_params_.items()},
               "cv_rmse_tuned_gb": round(-gb.best_score_, 3),
               "top_features": imp_s.tail(8)[::-1].round(3).to_dict(),
               "segment_error": seg_err.reset_index().to_dict("records"),
               "hotel_error": hotel_err.reset_index().to_dict("records"),
               "monthly": monthly.reset_index().rename(columns={"arrival_date_month": "month"}).to_dict("records"),
               "business": business, "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(results).T.to_csv(OUT / "model_comparison.csv")
    print(json.dumps(business, indent=2)); print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true", help="smaller search for a fast run")
    main(ap.parse_args().quick)
