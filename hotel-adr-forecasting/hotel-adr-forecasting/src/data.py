"""Data loading, cleaning and feature engineering for the hotel ADR model."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hotel_bookings.csv"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
TARGET = "adr"  # Average Daily Rate (revenue per occupied room-night)


def make_synthetic(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Synthetic data mimicking the Hotel Booking Demand schema (used if the CSV is absent)."""
    rng = np.random.default_rng(seed)
    hotel = rng.choice(["City Hotel", "Resort Hotel"], n, p=[0.6, 0.4])
    month = rng.choice(MONTHS, n)
    m_idx = np.array([MONTHS.index(m) for m in month])
    seg = rng.choice(["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"], n,
                     p=[0.47, 0.20, 0.13, 0.10, 0.10])
    room = rng.choice(list("ACDEFG"), n, p=[0.55, 0.05, 0.2, 0.1, 0.05, 0.05])
    lead = rng.gamma(1.5, 60, n).astype(int)
    wk_n = rng.integers(0, 6, n)
    we_n = rng.integers(0, 3, n)
    adults = rng.choice([1, 2, 3], n, p=[0.2, 0.72, 0.08])
    children = rng.choice([0, 1, 2], n, p=[0.92, 0.06, 0.02])
    season = np.where(hotel == "Resort Hotel",
                      55 * np.exp(-((m_idx - 7.5) ** 2) / 6),   # strong summer peak
                      25 * np.exp(-((m_idx - 5.5) ** 2) / 12))  # milder spring/summer peak
    base = np.where(hotel == "Resort Hotel", 70, 85)
    seg_eff = pd.Series(seg).map({"Online TA": 12, "Offline TA/TO": -8, "Direct": 5,
                                  "Corporate": -15, "Groups": -20}).values
    room_eff = pd.Series(room).map({"A": 0, "C": 25, "D": 18, "E": 30, "F": 55, "G": 70}).values
    adr = (base + season + seg_eff + room_eff + 12 * (adults - 2) + 15 * children
           - 0.04 * np.minimum(lead, 300) + 6 * (we_n > 0) + rng.normal(0, 13, n))
    return pd.DataFrame({
        "hotel": hotel, "lead_time": lead, "arrival_date_year": rng.choice([2015, 2016, 2017], n),
        "arrival_date_month": month, "arrival_date_week_number": np.clip(m_idx * 4 + rng.integers(1, 5, n), 1, 53),
        "arrival_date_day_of_month": rng.integers(1, 29, n),
        "stays_in_weekend_nights": we_n, "stays_in_week_nights": wk_n,
        "adults": adults, "children": children, "market_segment": seg,
        "reserved_room_type": room,
        "deposit_type": rng.choice(["No Deposit", "Non Refund", "Refundable"], n, p=[0.88, 0.11, 0.01]),
        "customer_type": rng.choice(["Transient", "Contract", "Transient-Party", "Group"], n, p=[0.75, 0.04, 0.2, 0.01]),
        "is_repeated_guest": rng.choice([0, 1], n, p=[0.96, 0.04]),
        "total_of_special_requests": rng.choice([0, 1, 2, 3], n, p=[0.55, 0.28, 0.13, 0.04]),
        "required_car_parking_spaces": rng.choice([0, 1], n, p=[0.93, 0.07]),
        "is_canceled": rng.choice([0, 1], n, p=[0.63, 0.37]),
        TARGET: adr,
    })


def load_raw() -> tuple[pd.DataFrame, str]:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), "kaggle"
    return make_synthetic(), "synthetic"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage columns, impossible rows and extreme ADR outliers."""
    df = df.copy()
    df = df.drop(columns=[c for c in ["reservation_status", "reservation_status_date"] if c in df.columns])
    df["children"] = df["children"].fillna(0)
    df = df[(df[TARGET] > 0) & (df[TARGET] < 500)]           # drop free stays & extreme outliers
    df = df[(df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) > 0]
    df = df[(df["adults"] + df["children"]) > 0]
    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features used by the optimised model."""
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"]
    df["month_num"] = df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)   # cyclical seasonality
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["has_weekend"] = (df["stays_in_weekend_nights"] > 0).astype(int)
    df["log_lead_time"] = np.log1p(df["lead_time"])
    df["has_children"] = (df["children"] > 0).astype(int)
    df["peak_season"] = df["month_num"].isin([6, 7, 8]).astype(int)
    return df
