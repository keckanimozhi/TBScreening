"""Mock GIS surveillance data for the admin dashboard (Task 12).

No real geolocated screening data exists yet -- the proposal's actual
surveillance dashboard would be populated by real screening events from
the mobile app during the Phase II/III clinical validation. This module
generates a deterministic (seeded) synthetic dataset of screening events
clustered around Perundurai, Erode (the proposal's primary clinical
site), purely so the dashboard UI/plumbing can be built and demoed now.
Every value here is fabricated -- do not present it as real surveillance
data.
"""

import numpy as np
import pandas as pd

PERUNDURAI_LAT = 11.2751
PERUNDURAI_LON = 77.5847

# A few nearby settlement clusters, loosely fanned out around the RTS
# Sanatorium / Perundurai (proposal's primary clinical hub), so the map
# shows a plausible hotspot pattern rather than a uniform random cloud.
CLUSTERS = [
    {"name": "Perundurai town", "lat": PERUNDURAI_LAT, "lon": PERUNDURAI_LON, "weight": 0.40},
    {"name": "Erode town", "lat": 11.3410, "lon": 77.7172, "weight": 0.25},
    {"name": "Kangayam road", "lat": 11.2200, "lon": 77.5600, "weight": 0.15},
    {"name": "Chennimalai", "lat": 11.1800, "lon": 77.6100, "weight": 0.20},
]

RISK_BAND_COLORS = {
    "Low": [16, 160, 90, 160],
    "Medium": [230, 160, 20, 160],
    "High": [210, 40, 40, 180],
}


def generate_mock_events(n: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cluster_idx = rng.choice(len(CLUSTERS), size=n, p=[c["weight"] for c in CLUSTERS])
    lats, lons = [], []
    for idx in cluster_idx:
        c = CLUSTERS[idx]
        lats.append(c["lat"] + rng.normal(0, 0.035))
        lons.append(c["lon"] + rng.normal(0, 0.035))

    # 12 months ending "now" for this mock dataset, weighted so recent
    # months have somewhat more cases (arbitrary, just for a non-flat
    # trend line -- not modeling any real seasonality).
    months = pd.period_range(end=pd.Timestamp.today(), periods=12, freq="M")
    month_weights = np.linspace(0.6, 1.4, num=12)
    month_weights /= month_weights.sum()
    event_months = rng.choice(months, size=n, p=month_weights)

    risk_probs = rng.beta(2, 3, size=n)  # skewed toward lower risk, long tail up
    risk_band = np.where(risk_probs >= 0.7, "High", np.where(risk_probs >= 0.3, "Medium", "Low"))

    df = pd.DataFrame(
        {
            "case_id": [f"MOCK-{i+1:04d}" for i in range(n)],
            "lat": lats,
            "lon": lons,
            "month": event_months.astype(str),
            "risk_score": risk_probs.round(3),
            "risk_band": risk_band,
        }
    )
    df["color"] = df["risk_band"].map(RISK_BAND_COLORS)
    return df


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("month").size().rename("cases").reset_index()
    return counts.sort_values("month")
