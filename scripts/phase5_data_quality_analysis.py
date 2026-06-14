"""
PHASE 5: DATA QUALITY & ANOMALY DETECTION
Veri kalitesi: eksik deger, outlier, duplicate, tutarsizlik.
"""

import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\sence\OneDrive\Desktop\churn-analysis")

PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
    "#BC4B51", "#8E7DBE", "#F77F00", "#06A77D", "#D4A574"
]

data_prep_recommendations = []

def add_rec(issue, evidence, recommendation, priority="Orta"):
    data_prep_recommendations.append({
        "Sorun": issue, "Kanit": evidence,
        "Oneri": recommendation, "Oncelik": priority
    })

def apply_premium_layout(fig, title, height=500):
    fig.update_layout(
        title={"text": title, "x": 0.03, "xanchor": "left",
               "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
        template="plotly_white", paper_bgcolor="#FBFBF8", plot_bgcolor="#FBFBF8",
        font={"family": "Arial", "size": 13, "color": "#374151"},
        margin=dict(l=60, r=40, t=90, b=60), height=height,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig

def save_figure(fig, file_base):
    html_path = PROJECT_ROOT / "figures" / f"{file_base}.html"
    png_path  = PROJECT_ROOT / "figures" / f"{file_base}.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path))
        print(f"  PNG: figures/{file_base}.png")
    except Exception as e:
        print(f"  PNG hatasi: {e}")
    return html_path, png_path

# ─────────────────────────────────────────────
df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "churn.csv")
NUMERIC_COLS = ["CreditScore", "Age", "Tenure", "Balance",
                "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CONT_COLS    = ["CreditScore", "Age", "Balance", "EstimatedSalary"]

print("=" * 60)
print("PHASE 5: VERI KALITESI & ANOMALI TESPITI")
print("=" * 60)

quality_records = []

# ─────────────────────────────────────────────
# 1. EKSİK VERİ
# ─────────────────────────────────────────────
print("\n--- Eksik Veri Analizi ---")
missing_count = df.isnull().sum()
missing_ratio = (missing_count / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Eksik_Sayi": missing_count, "Eksik_Oran_%": missing_ratio
}).sort_values("Eksik_Oran_%", ascending=False)

total_missing = missing_count.sum()
print(f"  Toplam eksik deger: {total_missing}")
if total_missing == 0:
    print("  SONUC: Eksik deger BULUNAMADI. Veri seti tamam.")
    quality_records.append({"Kontrol": "Eksik Veri", "Durum": "Temiz", "Detay": "Hicbir degiskende eksik veri yok."})
else:
    eksik_alan = missing_df[missing_df["Eksik_Oran_%"] > 0]
    print(eksik_alan.to_string())
    add_rec(
        issue="Eksik veri tespit edildi",
        evidence=f"Toplam {total_missing} eksik deger. Etkilenen degiskenler: {list(eksik_alan.index)}",
        recommendation="Imputation veya degisken cikarma degerlendirin.",
        priority="Yuksek"
    )

# ─────────────────────────────────────────────
# 2. MÜKERRER KAYIT
# ─────────────────────────────────────────────
print("\n--- Mukerrer Kayit Analizi ---")
dup_full  = df.duplicated().sum()
dup_cust  = df["CustomerId"].duplicated().sum()
dup_row   = df["RowNumber"].duplicated().sum()
print(f"  Tam satir tekrari      : {dup_full}")
print(f"  CustomerId tekrari     : {dup_cust}")
print(f"  RowNumber tekrari      : {dup_row}")

if dup_full > 0:
    add_rec("Mukerrer kayit", f"{dup_full} tam tekrar satir.", "Duplicate satirlari sil.", "Yuksek")
    quality_records.append({"Kontrol": "Mukerrer Kayit", "Durum": "Riskli", "Detay": f"{dup_full} duplicate satir."})
else:
    quality_records.append({"Kontrol": "Mukerrer Kayit", "Durum": "Temiz", "Detay": "Mukerrer satir yok."})

# ─────────────────────────────────────────────
# 3. IQR OUTLIER ANALİZİ
# ─────────────────────────────────────────────
print("\n--- IQR Outlier Analizi ---")
outlier_records = []

for col in NUMERIC_COLS:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - 1.5 * iqr
    hi  = q3 + 1.5 * iqr
    out_n = ((df[col] < lo) | (df[col] > hi)).sum()
    out_r = out_n / len(df) * 100

    outlier_records.append({
        "Degisken": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
        "IQR": round(iqr, 2), "Alt_Sinir": round(lo, 2), "Ust_Sinir": round(hi, 2),
        "Outlier_Sayi": out_n, "Outlier_Oran_%": round(out_r, 2)
    })
    print(f"  {col:20s}: {out_n:4d} outlier (%{out_r:.2f})")
    if out_r > 5:
        add_rec(
            issue=f"Yuksek outlier: {col}",
            evidence=f"%{out_r:.2f} outlier (IQR).",
            recommendation="Winsorization veya robust scaler onerilir.",
            priority="Orta"
        )

outlier_df = pd.DataFrame(outlier_records)
outlier_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase5_outlier_analysis.csv", index=False)

# Outlier bar chart
fig = px.bar(
    outlier_df.sort_values("Outlier_Oran_%", ascending=True),
    x="Outlier_Oran_%", y="Degisken",
    orientation="h",
    color="Outlier_Oran_%",
    color_continuous_scale=[[0, "#D5F5E3"], [0.4, "#F7D9A3"], [1.0, "#C73E1D"]],
    text=outlier_df.sort_values("Outlier_Oran_%")["Outlier_Oran_%"].apply(lambda x: f"%{x:.2f}"),
    title="IQR Yontemiyle Outlier Oranlari"
)
fig = apply_premium_layout(fig, "IQR Yontemiyle Outlier Oranlari", height=480)
fig.update_traces(textposition="outside")
fig.update_layout(xaxis_title="Outlier Orani (%)", yaxis_title="Degisken", coloraxis_showscale=False)
save_figure(fig, "phase5_outlier_ratios_iqr")

# ─────────────────────────────────────────────
# 4. Z-SCORE OUTLIER (SUREKLI DEGİSKENLER)
# ─────────────────────────────────────────────
print("\n--- Z-Score Outlier Analizi (|z| > 3) ---")
zscore_records = []
for col in CONT_COLS:
    z_scores  = np.abs(stats.zscore(df[col].dropna()))
    out_n     = (z_scores > 3).sum()
    out_r     = out_n / len(df) * 100
    zscore_records.append({
        "Degisken": col, "Outlier_Sayi_Zscore": out_n, "Outlier_Oran_%_Zscore": round(out_r, 2)
    })
    print(f"  {col:20s}: {out_n:4d} (%{out_r:.2f})")

zscore_df = pd.DataFrame(zscore_records)
zscore_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase5_zscore_outliers.csv", index=False)

# ─────────────────────────────────────────────
# 5. MANTIK KONTROLLERİ
# ─────────────────────────────────────────────
print("\n--- Mantik / Tutarlilik Kontrolleri ---")

# Yas araligi
age_out_of_range = ((df["Age"] < 18) | (df["Age"] > 100)).sum()
print(f"  Yas < 18 veya > 100    : {age_out_of_range} kayit")

# Negatif bakiye
neg_balance = (df["Balance"] < 0).sum()
print(f"  Negatif Balance        : {neg_balance} kayit")

# CreditScore siniri
cs_out = ((df["CreditScore"] < 300) | (df["CreditScore"] > 900)).sum()
print(f"  CreditScore [300-900]  : {cs_out} sinir disi kayit")

# NumOfProducts: sadece 1-4 olmali
nop_invalid = (~df["NumOfProducts"].isin([1,2,3,4])).sum()
print(f"  NumOfProducts gecersiz : {nop_invalid} kayit")

# Gender kategorileri
print(f"  Gender kategorileri    : {df['Gender'].unique()}")

# Geography kategorileri
print(f"  Geography kategorileri : {df['Geography'].unique()}")

quality_records += [
    {"Kontrol": "Yas Siniri", "Durum": "Temiz" if age_out_of_range == 0 else "Riskli",
     "Detay": f"{age_out_of_range} anormal yas"},
    {"Kontrol": "Negatif Balance", "Durum": "Temiz" if neg_balance == 0 else "Riskli",
     "Detay": f"{neg_balance} negatif bakiye"},
    {"Kontrol": "CreditScore Siniri", "Durum": "Temiz" if cs_out == 0 else "Riskli",
     "Detay": f"{cs_out} sinir disi kredi skoru"},
    {"Kontrol": "NumOfProducts", "Durum": "Temiz" if nop_invalid == 0 else "Riskli",
     "Detay": f"{nop_invalid} gecersiz deger"},
]

# ─────────────────────────────────────────────
# 6. CREDITSCORE DAGILIMI — OUTLIER VİZUALİZASYON
# ─────────────────────────────────────────────
fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=["<b>CreditScore Dagilimi</b>", "<b>Age Dagilimi</b>"],
    horizontal_spacing=0.12)

for idx, col in enumerate(["CreditScore", "Age"], 1):
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - 1.5 * iqr
    hi  = q3 + 1.5 * iqr
    out_mask = (df[col] < lo) | (df[col] > hi)

    fig2.add_trace(
        go.Scatter(x=df.index[~out_mask], y=df[col][~out_mask],
                   mode="markers", marker=dict(color=PROFESSIONAL_PALETTE[0], size=3, opacity=0.4),
                   name="Normal", showlegend=(idx == 1)),
        row=1, col=idx
    )
    fig2.add_trace(
        go.Scatter(x=df.index[out_mask], y=df[col][out_mask],
                   mode="markers", marker=dict(color="#C73E1D", size=6, symbol="x"),
                   name="Outlier", showlegend=(idx == 1)),
        row=1, col=idx
    )
    fig2.add_hline(y=lo, line_dash="dash", line_color="#F18F01", opacity=0.7, row=1, col=idx)
    fig2.add_hline(y=hi, line_dash="dash", line_color="#F18F01", opacity=0.7, row=1, col=idx)
    fig2.update_yaxes(title_text=col, row=1, col=idx, showgrid=True, gridcolor="#E5E7EB")
    fig2.update_xaxes(title_text="Indeks", row=1, col=idx)

fig2.update_layout(
    title={"text": "IQR Outlier Tespiti: CreditScore ve Age",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", height=480,
    font={"family": "Arial", "size": 13, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60),
    legend=dict(x=0.47, y=0.98)
)
save_figure(fig2, "phase5_outlier_scatter_creditscore_age")

# ─────────────────────────────────────────────
# 7. BALANCE YOGUNLUK: SIFIR BAKIYE
# ─────────────────────────────────────────────
zero_bal = (df["Balance"] == 0).sum()
nonzero_bal = (df["Balance"] > 0).sum()
print(f"\n  Balance=0 sayisi: {zero_bal} (%{zero_bal/len(df)*100:.1f})")
print(f"  Balance>0 sayisi: {nonzero_bal} (%{nonzero_bal/len(df)*100:.1f})")

add_rec(
    issue="Balance=0 yogunlugu",
    evidence=f"Musterilerin %{zero_bal/len(df)*100:.1f}'i sifir bakiyeye sahip ({zero_bal} kayit).",
    recommendation="has_zero_balance ikili flag degiskeni olusturulmasi Feature Engineering acisindan faydali.",
    priority="Orta"
)

# ─────────────────────────────────────────────
# 8. VERI KALİTESİ ÖZET TABLOSU
# ─────────────────────────────────────────────
quality_df = pd.DataFrame(quality_records)
quality_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase5_data_quality_summary.csv", index=False)

print("\n--- Veri Kalitesi Ozet ---")
print(quality_df.to_string(index=False))

# ─────────────────────────────────────────────
# KAYIT
# ─────────────────────────────────────────────
recs_df = pd.DataFrame(data_prep_recommendations) if data_prep_recommendations else pd.DataFrame(
    columns=["Sorun", "Kanit", "Oneri", "Oncelik"])
recs_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase5_data_prep_recommendations.csv", index=False)

print(f"\nPhase 5 tamamlandi. {len(data_prep_recommendations)} Data Prep onerisi kaydedildi.")
for r in data_prep_recommendations:
    print(f"  [{r['Oncelik']}] {r['Sorun']}")
