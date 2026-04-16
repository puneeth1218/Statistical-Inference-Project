import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =========================
# 1. LOAD & MERGE DATA
# =========================

df = pd.read_csv("combined_dataset.csv")


# =========================
# 2. CLEANING
# =========================

df = df.drop(columns=['Trip'], errors='ignore')

df = df.dropna(subset=[
    'Vehicle_Speed_km_per_h',
    'MAF_g_per_sec',
    'Engine_RPM_RPM',
    'EngineType'
])

# Reduce size (optional)
df = df.sample(frac=0.3, random_state=42)

# =========================
# 3. FEATURE CREATION
# =========================

df['Driving_Style'] = np.where(
    (df['Engine_RPM_RPM'] > 2000) |
    (df['MAF_g_per_sec'] > df['MAF_g_per_sec'].mean()),
    'Aggressive',
    'Smooth'
)

# =========================
# 4. FILTER DATA
# =========================

ice = df[df['EngineType'] == 'ICE']['MAF_g_per_sec']
hev = df[df['EngineType'] == 'HEV']['MAF_g_per_sec']
phev = df[df['EngineType'] == 'PHEV']['MAF_g_per_sec']

agg = df[df['Driving_Style'] == 'Aggressive']['MAF_g_per_sec']
smooth = df[df['Driving_Style'] == 'Smooth']['MAF_g_per_sec']

# =========================
# FUNCTIONS
# =========================

def compute_stats(data):
    return np.mean(data), np.var(data, ddof=1), np.std(data, ddof=1), len(data)

def confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    t_val = stats.t.ppf(0.975, df=n-1)
    margin = t_val * (std / np.sqrt(n))
    return (mean - margin, mean + margin)

def print_stats(name, data):
    mean, var, std, n = compute_stats(data)
    print(f"{name} → Mean: {mean:.2f}, Var: {var:.2f}, Std: {std:.2f}, n: {n}")
    print(f"{name} CI: {confidence_interval(data)}\n")

# =========================
# 🔬 A) ICE vs HEV vs PHEV
# =========================

print("\n===== ICE vs HEV vs PHEV =====")

print_stats("ICE", ice)
print_stats("HEV", hev)
print_stats("PHEV", phev)

# ---- Hypothesis 1: ICE > HEV ----
t_stat, p_val = stats.ttest_ind(ice, hev, equal_var=False)
p_one = p_val/2 if t_stat > 0 else 1 - (p_val/2)

print("\nH0: μICE = μHEV")
print("H1: μICE > μHEV")
print(f"t-stat: {t_stat:.2f}, one-sided p-value: {p_one:.5f}")

# ---- Hypothesis 2: HEV > PHEV ----
t_stat, p_val = stats.ttest_ind(hev, phev, equal_var=False)
p_one = p_val/2 if t_stat > 0 else 1 - (p_val/2)

print("\nH0: μHEV = μPHEV")
print("H1: μHEV > μPHEV")
print(f"t-stat: {t_stat:.2f}, one-sided p-value: {p_one:.5f}")

# =========================
# 🔬 B) DRIVING STYLE
# =========================

print("\n===== DRIVING STYLE =====")

print_stats("Aggressive", agg)
print_stats("Smooth", smooth)

t_stat, p_val = stats.ttest_ind(agg, smooth, equal_var=False)
p_one = p_val/2 if t_stat > 0 else 1 - (p_val/2)

print("\nH0: μAggressive = μSmooth")
print("H1: μAggressive > μSmooth")
print(f"t-stat: {t_stat:.2f}, one-sided p-value: {p_one:.5f}")

# =========================
# 🔬 C) CHI-SQUARE TEST (ROBUSTNESS)
# =========================

print("\n===== CHI-SQUARE TEST: Engine Type vs Consumption Level =====")

# ---- Create Consumption Categories (Quantile-based for robustness) ----
df['Consumption_Level'] = pd.qcut(
    df['MAF_g_per_sec'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# ---- Contingency Table ----
contingency_table = pd.crosstab(df['EngineType'], df['Consumption_Level'])

print("\nContingency Table:\n")
print(contingency_table)

# ---- Chi-Square Test ----
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Results:")
print(f"Chi2 Statistic: {chi2_stat:.2f}")
print(f"Degrees of Freedom: {dof}")
print(f"p-value: {p_val:.10f}")

# ---- Expected Frequencies ----
expected_df = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)

print("\nExpected Frequencies:\n")
print(expected_df)

# ---- Decision ----
alpha = 0.05
if p_val < alpha:
    print("\nDecision: Reject H0 → Engine type and consumption level are dependent")
else:
    print("\nDecision: Fail to reject H0")

# =========================
# 🔬 D) KS TEST (BONUS ROBUSTNESS)
# =========================

print("\n===== KS TEST (Distribution Comparison) =====")

ks_stat, ks_p = stats.ks_2samp(ice, hev)
print(f"ICE vs HEV → KS Stat: {ks_stat:.4f}, p-value: {ks_p:.10f}")

ks_stat, ks_p = stats.ks_2samp(hev, phev)
print(f"HEV vs PHEV → KS Stat: {ks_stat:.4f}, p-value: {ks_p:.10f}")

# =========================
# 📊 GRAPHS
# =========================

# ---- Engine Comparison ----
plt.figure(figsize=(7,5))
plt.boxplot([ice, hev, phev], tick_labels=['ICE', 'HEV', 'PHEV'])
plt.title("Fuel Consumption: ICE vs HEV vs PHEV")
plt.ylabel("MAF (g/sec)")
plt.grid(True)
plt.tight_layout()
plt.savefig("engine_comparison.png", dpi=300)
plt.show()

# ---- Mean Comparison ----
labels = ['ICE', 'HEV', 'PHEV']
means = [ice.mean(), hev.mean(), phev.mean()]

plt.figure(figsize=(7,5))
plt.bar(labels, means)
plt.title("Mean Fuel Consumption Comparison")
plt.ylabel("MAF (g/sec)")
plt.grid(True)
plt.tight_layout()
plt.savefig("engine_means.png", dpi=300)
plt.show()

# ---- Driving Style ----
plt.figure(figsize=(6,5))
plt.boxplot([agg, smooth], tick_labels=['Aggressive', 'Smooth'])
plt.title("Fuel Consumption by Driving Style")
plt.ylabel("MAF (g/sec)")
plt.grid(True)
plt.tight_layout()
plt.savefig("driving_style.png", dpi=300)
plt.show()