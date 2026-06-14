"""
PHASE 7: MODEL READINESS ASSESSMENT + FINAL REPORT GENERATION
"""

import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\sence\OneDrive\Desktop\churn-analysis")

PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
    "#BC4B51", "#8E7DBE", "#F77F00", "#06A77D", "#D4A574"
]

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

print("=" * 60)
print("PHASE 7: MODEL HAZIRLIK DEGERLENDIRMESI")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. MODEL HAZIRLIK KONTROL LISTESİ
# ─────────────────────────────────────────────
readiness_checks = [
    {
        "Kontrol": "Eksik Veri",
        "Durum": "Gecti",
        "Aciklama": "Hicbir degiskende eksik deger yok. Imputation gerekmiyor.",
        "Oncelik": "Dusuk"
    },
    {
        "Kontrol": "Mukerrer Kayit",
        "Durum": "Gecti",
        "Aciklama": "Tam satir tekrari yok. 195 CustomerId tekrari var, araştırılmalı.",
        "Oncelik": "Orta"
    },
    {
        "Kontrol": "ID Sutunlari",
        "Durum": "Gerekli",
        "Aciklama": "RowNumber, CustomerId, Surname modelden cikarilmali.",
        "Oncelik": "Yuksek"
    },
    {
        "Kontrol": "Kategorik Encoding",
        "Durum": "Gerekli",
        "Aciklama": "Geography (3 sinif) ve Gender (2 sinif) encode edilmeli.",
        "Oncelik": "Yuksek"
    },
    {
        "Kontrol": "Feature Scaling",
        "Durum": "Gerekli",
        "Aciklama": "CreditScore, Age, Balance, EstimatedSalary icin StandardScaler onerilir.",
        "Oncelik": "Yuksek"
    },
    {
        "Kontrol": "Sinif Dengesizligi",
        "Durum": "Gerekli",
        "Aciklama": "Exited=0: %79.7, Exited=1: %20.3. SMOTE veya class_weight='balanced' onerilir.",
        "Oncelik": "Yuksek"
    },
    {
        "Kontrol": "Outlier Islem",
        "Durum": "Dusuk Risk",
        "Aciklama": "Tum degiskenlerde outlier orani %1.3'un altinda. Islem gerekmeyebilir.",
        "Oncelik": "Dusuk"
    },
    {
        "Kontrol": "Veri Sizintisi Riski",
        "Durum": "Gecti",
        "Aciklama": "RowNumber ve CustomerId disinda leakage riski tasiyan degisken tespit edilmedi.",
        "Oncelik": "Dusuk"
    },
    {
        "Kontrol": "Train-Test Split",
        "Durum": "Gerekli",
        "Aciklama": "Sinif dengesi nedeniyle stratified split (stratify=y) zorunludur.",
        "Oncelik": "Yuksek"
    },
    {
        "Kontrol": "Feature Engineering",
        "Durum": "Onerilen",
        "Aciklama": "age_group, has_zero_balance, is_high_risk_products, geo_risk_flag faydali olabilir.",
        "Oncelik": "Orta"
    },
    {
        "Kontrol": "Multicollinearity",
        "Durum": "Gecti",
        "Aciklama": "Hicbir degisken cifti arasinda |r| > 0.50 korelasyon bulunmadi.",
        "Oncelik": "Dusuk"
    },
]

ready_df = pd.DataFrame(readiness_checks)
ready_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase7_model_readiness_checklist.csv", index=False)

print("\nModel Hazirlik Kontrol Listesi:")
for r in readiness_checks:
    icon = "[OK]" if r["Durum"] == "Gecti" else "[!!]" if r["Durum"] in ["Gerekli", "Onerilen"] else "[--]"
    print(f"  {icon} [{r['Oncelik']:8s}] {r['Kontrol']:25s}: {r['Durum']:10s} | {r['Aciklama'][:70]}")

# ─────────────────────────────────────────────
# 2. MODEL HAZIRLIK GÖRSELİ
# ─────────────────────────────────────────────
durum_renk = {"Gecti": "#6A994E", "Gerekli": "#C73E1D", "Onerilen": "#F18F01", "Dusuk Risk": "#2E86AB"}
ready_df["Renk"] = ready_df["Durum"].map(durum_renk)
ready_df["Oncelik_Skor"] = ready_df["Oncelik"].map({"Yuksek": 3, "Orta": 2, "Dusuk": 1})
ready_sorted = ready_df.sort_values("Oncelik_Skor", ascending=True)

fig1 = go.Figure(go.Bar(
    x=ready_sorted["Oncelik_Skor"],
    y=ready_sorted["Kontrol"],
    orientation="h",
    marker_color=ready_sorted["Renk"],
    text=ready_sorted["Durum"],
    textposition="outside"
))
fig1 = apply_premium_layout(
    fig1, "Model Hazirlik Degerlendirmesi: Kontrol Listesi", height=560
)
fig1.update_layout(
    xaxis=dict(tickvals=[1, 2, 3], ticktext=["Dusuk", "Orta", "Yuksek"],
               range=[0, 4], showgrid=True, gridcolor="#E5E7EB"),
    xaxis_title="Oncelik Seviyesi",
    yaxis_title="Kontrol Maddesi"
)
save_figure(fig1, "phase7_model_readiness_checklist")

# ─────────────────────────────────────────────
# 3. EN GUÇLÜ DEGISKENLER ÖZETI GÖRSELİ
# ─────────────────────────────────────────────
top_features = {
    "Age":            0.285,
    "IsActiveMember": -0.175,
    "Geography":      0.102,
    "NumOfProducts":  0.088,
    "Gender":         0.060,
    "Balance":        0.037,
    "HasCrCard":     -0.017,
    "CreditScore":   -0.014,
    "EstimatedSalary":-0.011,
    "Tenure":        -0.004,
}
feat_series = pd.Series(top_features).sort_values()
colors = ["#2E86AB" if v < 0 else "#C73E1D" for v in feat_series.values]

fig2 = go.Figure(go.Bar(
    x=feat_series.values, y=feat_series.index,
    orientation="h", marker_color=colors,
    text=[f"{v:+.3f}" for v in feat_series.values],
    textposition="outside"
))
fig2 = apply_premium_layout(fig2, "Degiskenlerin Churn ile Pearson Korelasyonu (Model Onemi)", height=520)
fig2.update_layout(
    xaxis=dict(range=[-0.30, 0.35], showgrid=True, gridcolor="#E5E7EB",
               zeroline=True, zerolinecolor="#374151", zerolinewidth=2),
    xaxis_title="Pearson r",
    yaxis_title="Degisken"
)
save_figure(fig2, "phase7_feature_correlation_final")

# ─────────────────────────────────────────────
# 4. VERİ KALİTESİ SKORKART
# ─────────────────────────────────────────────
scorecard = {
    "Eksik Veri": "10/10",
    "Mukerrer Kayit": "9/10",
    "Outlier Riski": "9/10",
    "Mantik Tutarlilik": "10/10",
    "Sinif Dengesi": "5/10",
    "Feature Kalitesi": "7/10",
    "Genel Hazirlik": "8/10",
}
sc_df = pd.DataFrame(list(scorecard.items()), columns=["Kriter", "Skor"])
sc_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase7_data_quality_scorecard.csv", index=False)

print("\n--- Veri Kalitesi Skorkart ---")
for k, v in scorecard.items():
    print(f"  {k:25s}: {v}")

print("\n--- SONUC: MODEL HAZIRLIK KARARI ---")
print("  Durum : KISMEN HAZIR")
print("  Neden : Veri kalitesi yuksek, eksik deger yok, outlier dusuk.")
print("          ANCAK: sinif dengesizligi, encoding ve scaling gerekmektedir.")
print("  Tavsiye: Data Prep Expert adimlarindan sonra modellemeye gecilebilir.")

print("\nPhase 7 tamamlandi.")
