"""
PHASE 4: MULTIVARIATE ANALYSIS
Korelasyon matrisi, pairplot, VIF analizi.
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

# Encoding: kategorik -> sayisal (korelasyon icin)
df_enc = df.copy()
df_enc["Geography_num"] = df_enc["Geography"].map({"France": 0, "Spain": 1, "Germany": 2})
df_enc["Gender_num"]    = df_enc["Gender"].map({"Male": 0, "Female": 1})

drop_cols = ["RowNumber", "CustomerId", "Surname", "Geography", "Gender"]
df_enc = df_enc.drop(columns=drop_cols)

print("=" * 60)
print("PHASE 4: COK DEGİSKENLI ANALIZ")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. KORELASYON MATRİSİ
# ─────────────────────────────────────────────
print("\n--- Pearson Korelasyon Matrisi ---")
corr = df_enc.corr(method="pearson")

print("\nExited ile korelasyonlar (buyukten kucuge):")
exited_corr = corr["Exited"].drop("Exited").sort_values(key=abs, ascending=False)
for col, val in exited_corr.items():
    bar = "#" * int(abs(val) * 30)
    sign = "+" if val > 0 else "-"
    print(f"  {col:20s}: {sign}{abs(val):.4f}  {bar}")

# Yüksek korelasyon kontrolü (Exited haric)
print("\nDegiskenler arasi yuksek korelasyon (|r| > 0.50, Exited haric):")
high_corr_pairs = []
cols = [c for c in corr.columns if c != "Exited"]
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        r = corr.loc[cols[i], cols[j]]
        if abs(r) > 0.50:
            print(f"  {cols[i]:20s} x {cols[j]:20s}: {r:.4f}")
            high_corr_pairs.append((cols[i], cols[j], round(r, 4)))
            if abs(r) > 0.80:
                add_rec(
                    issue=f"Yuksek korelasyon: {cols[i]} x {cols[j]}",
                    evidence=f"Pearson r = {r:.4f} (esik: 0.80).",
                    recommendation="VIF kontrolu ve feature selection onerilir.",
                    priority="Yuksek"
                )

if not high_corr_pairs:
    print("  Esik degerin (0.50) uzerinde anlamli korelasyon çifti bulunamadi.")

# ─── Isil harita (heatmap) ───
fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale=[
        [0.0,  "#C73E1D"],
        [0.25, "#F18F01"],
        [0.5,  "#FBFBF8"],
        [0.75, "#2E86AB"],
        [1.0,  "#1A3A52"]
    ],
    zmid=0,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    textfont={"size": 11, "color": "#1F2937"},
    hoverongaps=False,
    colorbar=dict(title="Korelasyon", tickfont=dict(size=11))
))
fig = apply_premium_layout(
    fig, "Pearson Korelasyon Matrisi (Tum Sayisal Degiskenler)", height=700
)
fig.update_layout(
    xaxis=dict(tickangle=-40, tickfont=dict(size=12)),
    yaxis=dict(tickfont=dict(size=12))
)
save_figure(fig, "phase4_correlation_matrix")

# ─────────────────────────────────────────────
# 2. EXITED İLE KORELASYON BAR CHART
# ─────────────────────────────────────────────
exited_corr_df = exited_corr.reset_index()
exited_corr_df.columns = ["Degisken", "Korelasyon"]
exited_corr_df = exited_corr_df.sort_values("Korelasyon", ascending=True)
exited_corr_df["Renk"] = exited_corr_df["Korelasyon"].apply(
    lambda x: "#C73E1D" if x > 0 else "#2E86AB"
)

fig2 = go.Figure(go.Bar(
    x=exited_corr_df["Korelasyon"],
    y=exited_corr_df["Degisken"],
    orientation="h",
    marker_color=exited_corr_df["Renk"],
    text=exited_corr_df["Korelasyon"].round(3),
    textposition="outside"
))
fig2 = apply_premium_layout(
    fig2, "Exited ile Pearson Korelasyonlari (Buyukten Kucuge)", height=520
)
fig2.update_layout(
    xaxis_title="Pearson Korelasyon Katsayisi",
    yaxis_title="Degisken",
    xaxis=dict(range=[-0.45, 0.45], showgrid=True, gridcolor="#E5E7EB", zeroline=True,
               zerolinecolor="#374151", zerolinewidth=2),
    yaxis=dict(showgrid=False)
)
save_figure(fig2, "phase4_exited_correlations")

# ─────────────────────────────────────────────
# 3. SCATTER MATRIX (PAIRPLOT) — KRITIK DEGİSKENLER
# ─────────────────────────────────────────────
key_cols = ["Age", "Balance", "CreditScore", "NumOfProducts", "EstimatedSalary", "Exited"]
df_pair = df_enc[key_cols].copy()
df_pair["Durum"] = df_pair["Exited"].map({0: "Kaldi", 1: "Ayrildi"})

fig3 = px.scatter_matrix(
    df_pair,
    dimensions=["Age", "Balance", "CreditScore", "NumOfProducts", "EstimatedSalary"],
    color="Durum",
    color_discrete_map={"Kaldi": "#2E86AB", "Ayrildi": "#C73E1D"},
    opacity=0.4
)
fig3.update_traces(diagonal_visible=False, marker=dict(size=3))
fig3 = apply_premium_layout(
    fig3, "Scatter Matrix: Kritik Degiskenler (Churn Renklendirmesi)", height=750
)
fig3.update_layout(legend=dict(x=0.85, y=0.95))
save_figure(fig3, "phase4_scatter_matrix")

# ─────────────────────────────────────────────
# 4. YAS x BAKIYE x CHURN SCATTER
# ─────────────────────────────────────────────
df_scat = df.copy()
df_scat["Durum"] = df_scat["Exited"].map({0: "Kaldi", 1: "Ayrildi"})

fig4 = px.scatter(
    df_scat, x="Age", y="Balance", color="Durum",
    color_discrete_map={"Kaldi": "#2E86AB", "Ayrildi": "#C73E1D"},
    opacity=0.55, size_max=6,
    labels={"Age": "Yas", "Balance": "Bakiye", "Durum": "Durum"}
)
fig4 = apply_premium_layout(fig4, "Yas x Bakiye: Churn Dagilimi", height=520)
fig4.update_layout(
    xaxis_title="Yas", yaxis_title="Bakiye",
    legend=dict(x=0.02, y=0.98)
)
fig4.update_traces(marker=dict(size=5))
save_figure(fig4, "phase4_scatter_age_balance_churn")

# ─────────────────────────────────────────────
# 5. ISACTIVEMEMBER x GEOGRAPHY x CHURN HEATMAP
# ─────────────────────────────────────────────
pivot_data = df.pivot_table(values="Exited", index="Geography",
                            columns="IsActiveMember", aggfunc="mean") * 100
pivot_data.columns = ["Pasif Uye", "Aktif Uye"]
print("\n--- Cografi Bolge x Uyelik Durumu: Churn Orani (%) ---")
print(pivot_data.round(1).to_string())

fig5 = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=pivot_data.columns.tolist(),
    y=pivot_data.index.tolist(),
    colorscale=[[0, "#D5F5E3"], [0.5, "#F7D9A3"], [1.0, "#C73E1D"]],
    text=np.round(pivot_data.values, 1),
    texttemplate="%{text}%",
    textfont={"size": 16, "color": "#1F2937"},
    colorbar=dict(title="Churn %")
))
fig5 = apply_premium_layout(
    fig5, "Cografi Bolge x Uyelik Durumu: Churn Orani (%) Isil Harita", height=400
)
save_figure(fig5, "phase4_geo_activemember_churn_heatmap")

# ─────────────────────────────────────────────
# KAYIT
# ─────────────────────────────────────────────
corr.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase4_correlation_matrix.csv")

recs_df = pd.DataFrame(data_prep_recommendations) if data_prep_recommendations else pd.DataFrame(
    columns=["Sorun", "Kanit", "Oneri", "Oncelik"])
recs_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase4_data_prep_recommendations.csv", index=False)

print(f"\nPhase 4 tamamlandi.")
print(f"Yuksek korelasyon ciftleri (|r|>0.50): {len(high_corr_pairs)}")
print(f"Data Prep onerisi: {len(data_prep_recommendations)} adet")
