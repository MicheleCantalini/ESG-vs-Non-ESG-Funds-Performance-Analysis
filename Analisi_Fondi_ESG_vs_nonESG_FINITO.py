
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_ind

# === Caricamento file ===
df = pd.read_excel("Fondi_ESG_e_non_ESG_CORRETTO.xlsx")

# === Pulizia colonne ===
df["ESG"] = df["Is ESG Intentional Investment - Overall"].map({"Yes": 1, "No": 0})

# === Conversione a numerico ===
colonne = [
    "Total Ret Annlzd 3 Yr (Mo-End) Base Currency",
    "Total Ret Annlzd 5 Yr (Mo-End) Base Currency",
    "Total Ret Annlzd 10 Yr (Mo-End) Base Currency",
    "Equity Region Greater Europe % (Net)",
    "Equity Region Europe dev % (Net)",
    "Equity Region Europe emrg % (Net)"
]
df[colonne] = df[colonne].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=["ESG"] + colonne, inplace=True)

# === Statistiche descrittive ===
desc_stats = df.groupby("ESG")[colonne].describe().round(3)
print("=== Statistiche Descrittive per ESG vs Non ESG ===")
print(desc_stats)

# === T-test ===
for i, label in zip(range(3), ["3 anni", "5 anni", "10 anni"]):
    ttest = ttest_ind(df[df["ESG"] == 1][colonne[i]], df[df["ESG"] == 0][colonne[i]])
    print(f"\n=== T-test - {label} ===")
    print(f"Statistic = {ttest.statistic:.4f}, p-value = {ttest.pvalue:.4f}")

# === Regressioni lineari e multiple ===
def regressioni(y):
    X_lin = sm.add_constant(df["ESG"])
    model_lin = sm.OLS(df[y], X_lin).fit()
    X_multi = sm.add_constant(df[["ESG"] + colonne[3:]])
    model_multi = sm.OLS(df[y], X_multi).fit()
    return model_lin, model_multi

for i, label in zip(range(3), ["3Y", "5Y", "10Y"]):
    lin, multi = regressioni(colonne[i])
    print(f"\n=== Regressione Lineare ({label}) ===")
    print(lin.summary())
    print(f"\n=== Regressione Multipla ({label}) ===")
    print(multi.summary())

# === Performance metrics ===
def performance_metrics(df, col, label):
    results = []
    for group, data in df.groupby("ESG"):
        r = data[col]
        sharpe = r.mean() / r.std()
        sortino = r.mean() / r[r < 0].std()
        var95 = np.percentile(r, 5)
        results.append({
            "Periodo": label,
            "ESG": group,
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
            "VaR 95%": round(var95, 4)
        })
    return pd.DataFrame(results)

all_perf = pd.concat([
    performance_metrics(df, colonne[0], "3Y"),
    performance_metrics(df, colonne[1], "5Y"),
    performance_metrics(df, colonne[2], "10Y")
], ignore_index=True)

print("\n=== Metriche di Performance (Sharpe, Sortino, VaR) ===")
print(all_perf.to_string(index=False))
