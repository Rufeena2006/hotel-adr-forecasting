# Hotel ADR Forecasting - Model Evaluation, Optimization & Strategic Reporting

Week 5 internship project: evaluate and optimise a predictive model for hotel **Average Daily Rate (ADR)**,
then translate the results into business recommendations for hospitality decision makers.

## What it does
1. **Evaluate** baselines (mean predictor, Linear Regression, default Random Forest) with MAE, RMSE, R-squared, MAPE.
2. **Optimise** with feature engineering (cyclical seasonality, log lead time, stay/party features) and
   `RandomizedSearchCV` (5-fold CV) on Gradient Boosting and Random Forest.
3. **Interpret** with permutation importance, residual diagnostics and segment-level error analysis.
4. **Report** - business impact estimate + auto-generated Word report.

## Project structure
```
src/data.py          load/clean/engineer data (synthetic fallback if CSV missing)
src/modeling.py      models, metrics, tuning
run_pipeline.py      full pipeline -> outputs/metrics.json + outputs/figures/*.png
build_report.py      builds report/Week5_Model_Evaluation_Report.docx
report/              final Word report
outputs/             metrics, comparison table, figures
```

## Quick start
```bash
git clone <your-repo-url> && cd hotel-adr-forecasting
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # add --quick for a fast demo run
python build_report.py
```

## Data
Download **Hotel Booking Demand** from Kaggle and save as `data/hotel_bookings.csv`.
Without it, a schema-compatible synthetic dataset is generated so the code always runs.
Results in the committed report use the synthetic data - re-run with the real CSV to refresh them.

## Results (synthetic demo data)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 12.48 | 15.58 | 0.752 |
| Random Forest (default) | 11.63 | 14.59 | 0.783 |
| **Tuned Gradient Boosting** | **10.59** | **13.25** | **0.821** |

## Tech
Python 3.10+, pandas, scikit-learn, scipy, matplotlib, python-docx.
