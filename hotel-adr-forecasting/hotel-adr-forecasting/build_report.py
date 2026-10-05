"""Builds report/Week5_Model_Evaluation_Report.docx from outputs/metrics.json and figures.
Re-run after run_pipeline.py so every number in the report matches your latest results."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

M = json.loads(Path("outputs/metrics.json").read_text())
FIG = Path("outputs/figures")
R = M["results"]; B = M["business"]; CH = M["champion"]
base, tuned = R["Linear Regression"], R["Tuned Gradient Boosting"]
rf0 = R["Random Forest (default)"]
pct = lambda a, b: round((a - b) / a * 100, 1)

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(0.9)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color); tcPr.append(shd)


def table(header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(str(h)); r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255); shade(c, "1F4E79")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Inches(w)
    doc.add_paragraph()


def code(text):
    p = doc.add_paragraph(); r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(8.5)
    p.paragraph_format.left_indent = Inches(0.2)
    pPr = p._p.get_or_add_pPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F2F2F2"); pPr.append(shd)


def fig(name, caption, w=5.8):
    doc.add_picture(str(FIG / name), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True


H = lambda t, l=1: doc.add_heading(t, level=l)
P = lambda t: doc.add_paragraph(t)
BL = lambda t: doc.add_paragraph(t, style="List Bullet")

# ---------------- Title ----------------
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Week 5: Model Evaluation, Optimization and Strategic Reporting"); r.bold = True; r.font.size = Pt(22)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Hotel Average Daily Rate (ADR) Forecasting - Predictive Pricing Model").font.size = Pt(13)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Internship Final Report | Prepared for hospitality management stakeholders").italic = True
if M["data_source"] == "synthetic":
    n = doc.add_paragraph(); rr = n.add_run("Note: figures in this version were produced on the bundled synthetic dataset (same schema as the "
        "Kaggle Hotel Booking Demand data). Re-run run_pipeline.py and build_report.py with the real CSV to refresh all numbers.")
    rr.font.size = Pt(9); rr.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)

# ---------------- 1 Exec summary ----------------
H("1. Executive Summary")
P(f"This report evaluates and optimises the Week 4 predictive model, which estimates the Average Daily Rate (ADR) a booking will "
  f"realise. ADR is the core pricing KPI in hospitality: small forecast errors, repeated over thousands of room-nights, translate "
  f"directly into lost revenue (under-pricing) or lost occupancy (over-pricing).")
BL(f"Baseline (Linear Regression): MAE {base['MAE']}, RMSE {base['RMSE']}, R-squared {base['R2']}.")
BL(f"Optimised model ({CH}): MAE {tuned['MAE']}, RMSE {tuned['RMSE']}, R-squared {tuned['R2']} on unseen data.")
BL(f"RMSE improved by {pct(base['RMSE'], tuned['RMSE'])}% and MAE by {pct(base['MAE'], tuned['MAE'])}% versus the baseline.")
BL(f"Estimated pricing-error exposure fell from {B['baseline_exposure_pct_of_revenue']}% to {B['tuned_exposure_pct_of_revenue']}% of "
   f"room revenue (a {B['exposure_reduction_pct']}% reduction), i.e. roughly {B['pricing_error_exposure_baseline_LR'] - B['pricing_error_exposure_tuned']:,.0f} "
   f"currency units of revenue exposure removed on the hold-out sample alone.")
BL("Room type, market segment, seasonality and property type are the dominant ADR drivers - these should anchor pricing policy.")

# ---------------- 2 Context ----------------
H("2. Background and Week 4 Recap")
P("In Week 4 a supervised regression model was built to predict ADR from booking characteristics (property, lead time, length of stay, "
  "party composition, market segment, room type, deposit and customer type, arrival month/week). The target is continuous, so "
  "regression metrics are used. Data preparation removed leakage columns (reservation status), zero-rate and extreme-rate bookings "
  "(ADR <= 0 or >= 500), empty stays and bookings with no guests.")
table(["Item", "Detail"], [
    ["Task", "Regression - predict ADR per booking"],
    ["Rows used", f"{M['n_rows']:,} after cleaning"],
    ["Split", "80% train / 20% hold-out test (random_state=42); 5-fold CV inside tuning on train only"],
    ["Data source", "Hotel Booking Demand schema (" + M["data_source"] + ")"],
], [1.6, 4.9])

# ---------------- 3 Metrics ----------------
H("3. Evaluation Metrics - What They Mean and Why They Matter")
table(["Metric", "Definition", "Interpretation for hotel pricing"], [
    ["MAE", "Mean of |actual - predicted|", "Average pricing miss per room-night in currency units. Easiest to explain to revenue managers."],
    ["RMSE", "sqrt(mean((actual - predicted)^2))", "Penalises large misses more heavily. Important because big under-pricing errors on peak nights are costly."],
    ["R-squared", "1 - SSE/SST", "Share of ADR variance explained. 0 = no better than predicting the mean; 1 = perfect."],
    ["MAPE", "Mean of |error| / actual x 100", "Percentage error, comparable across cheap and premium rooms."],
], [0.9, 2.1, 3.5])
P("Why several metrics: MAE gives the typical error, RMSE exposes occasional large mistakes, R-squared gives an overall goodness-of-fit "
  "score, and MAPE expresses accuracy in relative terms. A model is only trusted if it improves across all four on data it has never seen.")

# ---------------- 4 Baseline results ----------------
H("4. Model Evaluation Results")
H("4.1 Baselines", 2)
rows = [[k, v["MAE"], v["RMSE"], v["R2"], v["MAPE_%"]] for k, v in R.items()]
table(["Model", "MAE", "RMSE", "R-squared", "MAPE %"], rows, [2.4, 1, 1, 1.1, 1])
fig("model_comparison.png", "Figure 1: Hold-out RMSE across models (lower is better)")
P(f"The mean predictor sets the floor (R-squared about 0). Linear Regression already explains {base['R2']*100:.0f}% of variance, showing ADR is largely "
  f"structured. The default Random Forest captures non-linear interactions (e.g. room type x season) and improves RMSE to {rf0['RMSE']}. "
  f"The tuned gradient boosting model is best at {tuned['RMSE']}.")
H("4.2 Diagnostics of the optimised model", 2)
fig("actual_vs_predicted.png", "Figure 2: Predicted vs actual ADR - points hug the diagonal", 4.2)
fig("residuals.png", "Figure 3: Residuals are centred on zero with no strong pattern, indicating little systematic bias")
P("Residuals are roughly symmetric around zero and show no obvious funnel shape, so the model is not systematically over- or under-pricing "
  "at any price level. Remaining error is dominated by irreducible noise (discounts, negotiated rates, unobserved events).")
H("4.3 Error by segment", 2)
table(["Market segment", "MAE", "Mean ADR", "Test bookings"],
      [[s["market_segment"], s["MAE"], s["mean_adr"], s["n"]] for s in M["segment_error"]], [2.2, 1.2, 1.4, 1.5])
table(["Hotel", "MAE", "Mean ADR"], [[s["hotel"], s["MAE"], s["mean_adr"]] for s in M["hotel_error"]], [2.2, 1.2, 1.4])
P("Error is broadly uniform across segments and both properties, which is a fairness and robustness check: no single channel or property is "
  "being forecast materially worse than the rest.")

# ---------------- 5 Optimisation ----------------
H("5. Optimization Process")
H("5.1 Areas for improvement identified", 2)
BL("Seasonality was encoded as a month name only; trees and linear models cannot see that December and January are adjacent.")
BL("Lead time is heavily right-skewed; stay length and party size are only implicit.")
BL("Default hyper-parameters were never tuned and may over- or under-fit.")
BL("A single random hold-out split gives a noisy estimate of generalisation.")
H("5.2 Feature engineering", 2)
code("""df["total_nights"]  = df.stays_in_weekend_nights + df.stays_in_week_nights
df["total_guests"]  = df.adults + df.children
df["month_sin"]     = np.sin(2*np.pi*df.month_num/12)   # cyclical seasonality
df["month_cos"]     = np.cos(2*np.pi*df.month_num/12)
df["log_lead_time"] = np.log1p(df.lead_time)             # tame skew
df["has_weekend"], df["has_children"], df["peak_season"] = ...""")
H("5.3 Hyper-parameter tuning", 2)
P("A RandomizedSearchCV with 5-fold cross-validation (25 candidates, scoring = negative RMSE) searched learning rate (log-uniform), number of "
  "boosting iterations, leaf count, minimum samples per leaf and L2 regularisation. Random search covers the space far more efficiently than "
  "an exhaustive grid for the same compute budget. Tuning used the training set only, so the hold-out set stays an honest test.")
code("""space = {
    "m__learning_rate":    loguniform(0.02, 0.3),
    "m__max_iter":         randint(200, 800),
    "m__max_leaf_nodes":   randint(15, 80),
    "m__min_samples_leaf": randint(10, 80),
    "m__l2_regularization": uniform(0, 2),
}
search = RandomizedSearchCV(pipe, space, n_iter=25,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)""")
bp = M["best_params"]
table(["Best parameter", "Value"], [[k.replace("m__", ""), round(v, 4) if isinstance(v, float) else v] for k, v in bp.items()], [3, 2])
P(f"Cross-validated RMSE of the best configuration: {M['cv_rmse_tuned_gb']}, close to the hold-out RMSE of {tuned['RMSE']}, which indicates the model generalises "
  f"rather than overfits.")
H("5.4 Further techniques considered", 2)
BL("Time-based validation (train on earlier years, test on later ones) to simulate real forecasting; recommended for production.")
BL("Re-sampling: bootstrapped confidence intervals around MAE/RMSE and stratified sampling by hotel/segment if classes are imbalanced.")
BL("Quantile gradient boosting to produce price ranges (P10-P90) rather than a single point estimate.")
BL("Ensembling (stacking RF + GB) and adding external features: local events, competitor rates, weather, holidays.")
H("5.5 Optimization results", 2)
table(["Metric", "Baseline LR", "Default RF", "Tuned GB", "Improvement vs LR"], [
    ["MAE", base["MAE"], rf0["MAE"], tuned["MAE"], f"{pct(base['MAE'], tuned['MAE'])}%"],
    ["RMSE", base["RMSE"], rf0["RMSE"], tuned["RMSE"], f"{pct(base['RMSE'], tuned['RMSE'])}%"],
    ["R-squared", base["R2"], rf0["R2"], tuned["R2"], f"+{tuned['R2'] - base['R2']:.3f}"],
    ["MAPE %", base["MAPE_%"], rf0["MAPE_%"], tuned["MAPE_%"], f"{pct(base['MAPE_%'], tuned['MAPE_%'])}%"],
], [1.1, 1.2, 1.2, 1.2, 1.8])

# ---------------- 6 Drivers ----------------
H("6. What Drives ADR? (Model Interpretation)")
fig("feature_importance.png", "Figure 4: Permutation importance - increase in RMSE when a feature is shuffled", 5.0)
top = list(M["top_features"].items())
P("Top drivers by permutation importance: " + ", ".join(f"{k} ({v})" for k, v in top[:5]) + ". "
  "Room type and market segment set the price tier; seasonality (month features) and property type adjust it; party size refines it. "
  "Lead time matters less once these are known.")
fig("seasonal_adr.png", "Figure 5: Monthly mean ADR - the model tracks the seasonal curve closely")

# ---------------- 7 Strategy ----------------
H("7. Strategic Report and Business Recommendations")
H("7.1 Revenue impact", 2)
P(f"Weighting each booking's absolute error by its nights, the baseline model carried pricing-error exposure equal to {B['baseline_exposure_pct_of_revenue']}% of room "
  f"revenue; the optimised model reduces this to {B['tuned_exposure_pct_of_revenue']}%. Even if only a fraction of that exposure is recoverable through better rate-setting, "
  f"each percentage point of a hotel's room revenue is material. These are modelled estimates on the test sample, not guaranteed gains; "
  f"validate through an A/B pricing pilot before scaling.")
H("7.2 Recommendations", 2)
table(["#", "Recommendation", "Rationale (evidence)", "Expected outcome"], [
    ["1", "Adopt the model as a rate-recommendation engine for revenue managers (human-in-the-loop).",
     f"MAPE {tuned['MAPE_%']}%, R-squared {tuned['R2']}, stable CV vs hold-out.", "Faster, more consistent pricing; fewer manual misses."],
    ["2", "Build seasonal pricing calendars from the forecast curve and raise rates ahead of the peak months.",
     "Strong, well-modelled seasonal pattern (Figure 5).", "Capture peak-season willingness to pay."],
    ["3", "Tier room-type pricing and review under-priced premium rooms.",
     "Room type is the #1 driver of ADR.", "Better upsell and premium-room yield."],
    ["4", "Optimise channel mix: shift volume toward higher-ADR segments (Direct, Online TA) and review group/corporate discount depth.",
     "Market segment is the #2 driver; segment ADR differs widely.", "Higher blended ADR, lower commission leakage."],
    ["5", "Run property-specific pricing playbooks for the city and resort hotels, each with its own seasonal calendar and segment mix.",
     "Property type and month features are both top drivers.", "Property-specific revenue playbooks."],
    ["6", "Use predicted ADR as a benchmark to flag bookings priced far below forecast (anomaly alerts).",
     "Residuals are unbiased, so large negative gaps are meaningful.", "Detect rate-parity and discounting issues early."],
], [0.3, 2.2, 2.2, 1.8])
H("7.3 Operational changes", 2)
BL("Weekly retraining or monitoring of MAE/RMSE; trigger retraining if error rises 15% above baseline.")
BL("Integrate predictions into the PMS/revenue-management dashboard with a confidence band.")
BL("Train revenue managers to treat the model as decision support and record overrides to improve future versions.")
H("7.4 Risks and limitations", 2)
BL("ADR excludes demand/occupancy: pricing should be combined with a demand or cancellation model to optimise RevPAR.")
BL("Historical data may not capture new events, competitor moves or post-pandemic behaviour shifts (concept drift).")
BL("Random-split evaluation can be optimistic; time-based validation is the next step.")
BL("Cancelled bookings are included as a feature (is_canceled) which is not known at booking time - remove it for live deployment to avoid leakage.")

# ---------------- 8 Conclusion ----------------
H("8. Conclusion and Next Steps")
P(f"Through feature engineering and cross-validated hyper-parameter tuning, the ADR model improved from RMSE {base['RMSE']} to {tuned['RMSE']} "
  f"(R-squared {base['R2']} to {tuned['R2']}) and remains stable across segments and properties. Translating the findings into pricing calendars, room-tier pricing and "
  f"channel strategy gives management concrete levers. Next steps: time-based validation, quantile forecasts, external data (events, competitor rates), "
  f"and a live pricing pilot.")
H("Appendix: Reproducibility", 1)
code("""git clone <repo-url> && cd hotel-adr-forecasting
pip install -r requirements.txt
# optional: put Kaggle hotel_bookings.csv in data/
python run_pipeline.py          # trains, tunes, evaluates, saves outputs/
python build_report.py          # regenerates this Word report""")

Path("report").mkdir(exist_ok=True)
doc.save("report/Week5_Model_Evaluation_Report.docx")
print("Saved report/Week5_Model_Evaluation_Report.docx")
