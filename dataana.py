# ============================================================
# Data Thinking Course
# Topic: China Trade Deficit (Jan–Nov)
# ------------------------------------------------------------
# This script:
# 1. Loads and cleans trade data
# 2. Runs four simple econometric models
# 3. Saves regression outputs and figures to ./output/
#
# Models included:
# (1) Trade balance driving model
# (2) Oil price & exchange rate explaining imports
# (3) Current-year (Jan–Nov) trend model
# (4) Historical Jan–Nov comparison model
#
# Note:
# These models are for data interpretation, not causal inference.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------------------------------------
# 0. File paths and folders
# ------------------------------------------------------------

DATA_PATH = "data.xlsx"          # raw data file
OUTPUT_DIR = "output"            # folder to store results

# create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load and preprocess data
# ------------------------------------------------------------

# read Excel data
df = pd.read_excel(DATA_PATH)

# convert date format: 2015.01 -> datetime
df["year"] = df["date"].astype(int)
df["month"] = ((df["date"] - df["year"]) * 100 + 1e-6).astype(int)
df["date_dt"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
)

# sort by time
df = df.sort_values("date_dt").reset_index(drop=True)

# ------------------------------------------------------------
# 2. Model 1: Trade balance driving model
# balance ~ export + import
# ------------------------------------------------------------

X1 = df[["export/usd", "import/usd"]]
X1 = sm.add_constant(X1)
y1 = df["balance/usd"]

model1 = sm.OLS(y1, X1).fit()

# save regression result
with open(os.path.join(OUTPUT_DIR, "model1_balance_driver.txt"), "w") as f:
    f.write(model1.summary().as_text())

# ------------------------------------------------------------
# 3. Model 2: Oil price & exchange rate → import
# import ~ oil_price + usd_cny
# ------------------------------------------------------------

X2 = df[["oil_price(usd/barrel)", "usd_cny"]]
X2 = sm.add_constant(X2)
y2 = df["import/usd"]

model2 = sm.OLS(y2, X2).fit()

# save regression result
with open(os.path.join(OUTPUT_DIR, "model2_import_price_fx.txt"), "w") as f:
    f.write(model2.summary().as_text())

# ------------------------------------------------------------
# 4. Model 3: Current-year (Jan–Nov) trend model
# balance ~ time
# ------------------------------------------------------------

latest_year = df["year"].max()

df_current = df[
    (df["year"] == latest_year) & (df["month"] <= 11)
].copy()

# time index
df_current["t"] = np.arange(1, len(df_current) + 1)

X3 = sm.add_constant(df_current["t"])
y3 = df_current["balance/usd"]

model3 = sm.OLS(y3, X3).fit()

# save regression result
with open(os.path.join(OUTPUT_DIR, "model3_current_year_trend.txt"), "w") as f:
    f.write(model3.summary().as_text())

# plot trend
plt.figure(figsize=(8, 4))
plt.plot(df_current["date_dt"], y3, label="Trade Balance")
plt.plot(
    df_current["date_dt"],
    model3.predict(X3),
    linestyle="--",
    label="Trend"
)
plt.title(f"{latest_year} Jan–Nov Trade Balance Trend")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model3_current_year_trend.png"))
plt.close()

# ------------------------------------------------------------
# 5. Model 4: Historical Jan–Nov comparison model
# annual_balance(1–11) ~ year
# ------------------------------------------------------------

df_1_11 = df[df["month"] <= 11]

annual = (
    df_1_11
    .groupby("year")[["export/usd", "import/usd", "balance/usd"]]
    .sum()
    .reset_index()
)

X4 = sm.add_constant(annual["year"])
y4 = annual["balance/usd"]

model4 = sm.OLS(y4, X4).fit()

# save regression result
with open(os.path.join(OUTPUT_DIR, "model4_historical_comparison.txt"), "w") as f:
    f.write(model4.summary().as_text())

# plot historical comparison
plt.figure(figsize=(8, 4))
plt.plot(annual["year"], y4, marker="o", label="Jan–Nov Balance")
plt.axhline(y=y4.mean(), linestyle="--", label="Historical Mean")
plt.title("Historical Comparison of Jan–Nov Trade Balance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model4_historical_comparison.png"))
plt.close()