"""
PHASE 2: UNIVARIATE ANALYSIS
Her degiskenin tek basina davranisinI anlamak.
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
        margin=dict(l=60, r=40, t=80, b=60), height=height,
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

# Gereksiz sutunlari cikar
drop_cols = ["RowNumber", "CustomerId", "Surname"]
df_model = df.drop(columns=drop_cols)

NUMERIC_COLS   = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                  "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CONT_COLS      = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
BINARY_COLS    = ["HasCrCard", "IsActiveMember"]
DISCRETE_COLS  = ["NumOfProducts"]
CAT_COLS       = ["Geography", "Gender"]

print("=" * 60)
print("PHASE 2: TEK DEGİSKENLI ANALIZ")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. SÜREKLI SAYISAL DEGİSKENLER
# ─────────────────────────────────────────────
print("\n--- Surekli Sayisal Degiskenler ---")

skew_records = []

for col in CONT_COLS:
    mean_  = df[col].mean()
    median_= df[col].median()
    std_   = df[col].std()
    skew_  = df[col].skew()
    kurt_  = df[col].kurtosis()
    q1     = df[col].quantile(0.25)
    q3     = df[col].quantile(0.75)
    iqr    = q3 - q1
    lo     = q1 - 1.5 * iqr
    hi     = q3 + 1.5 * iqr
    out_n  = ((df[col] < lo) | (df[col] > hi)).sum()
    out_r  = out_n / len(df) * 100

    skew_records.append({
        "Degisken": col, "Ortalama": round(mean_, 2),
        "Medyan": round(median_, 2), "Std": round(std_, 2),
        "Carpiklik": round(skew_, 3), "Basiklik": round(kurt_, 3),
        "Outlier_Sayi": out_n, "Outlier_Oran_%": round(out_r, 2)
    })
    print(f"  {col:20s} | carpiklik={skew_:.3f} | outlier={out_r:.1f}%")

    if abs(skew_) > 1:
        add_rec(
            issue=f"Yuksek carpiklik: {col}",
            evidence=f"{col} carpikligi {skew_:.3f} (|skewness| > 1).",
            recommendation="Log, Box-Cox veya Yeo-Johnson donusumu degerlendirilmeli.",
            priority="Orta"
        )
    if out_r > 5:
        add_rec(
            issue=f"Yuksek outlier orani: {col}",
            evidence=f"{col} outlier orani %{out_r:.2f} (IQR yontemi).",
            recommendation="Winsorization, robust scaler veya log donusumu onerilir.",
            priority="Orta"
        )

skew_df = pd.DataFrame(skew_records)
print("\n", skew_df.to_string(index=False))

# ─── Histogram + KDE panel (surekliler) ───
fig = make_subplots(rows=2, cols=3,
    subplot_titles=[f"<b>{c}</b>" for c in CONT_COLS],
    horizontal_spacing=0.10, vertical_spacing=0.18)

row_col = [(1,1),(1,2),(1,3),(2,1),(2,2)]
for idx, col in enumerate(CONT_COLS):
    r, c = row_col[idx]
    color = PROFESSIONAL_PALETTE[idx]
    fig.add_trace(
        go.Histogram(x=df[col], nbinsx=50,
                     marker_color=color, opacity=0.85,
                     name=col, showlegend=False),
        row=r, col=c
    )
    fig.update_xaxes(title_text=col, row=r, col=c,
                     showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(title_text="Frekans", row=r, col=c,
                     showgrid=True, gridcolor="#E5E7EB")

fig.update_layout(
    title={"text": "Surekli Degiskenler: Dagilim Histogramlari",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", plot_bgcolor="#FBFBF8",
    height=600, font={"family": "Arial", "size": 12, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60)
)
save_figure(fig, "phase2_histograms_continuous")

# ─── Boxplot paneli ───
fig2 = make_subplots(rows=1, cols=5,
    subplot_titles=[f"<b>{c}</b>" for c in CONT_COLS],
    horizontal_spacing=0.08)

for idx, col in enumerate(CONT_COLS, 1):
    fig2.add_trace(
        go.Box(y=df[col], name=col,
               marker_color=PROFESSIONAL_PALETTE[idx-1],
               line_color=PROFESSIONAL_PALETTE[idx-1],
               fillcolor=PROFESSIONAL_PALETTE[idx-1],
               opacity=0.8, showlegend=False),
        row=1, col=idx
    )
    fig2.update_yaxes(title_text=col, row=1, col=idx,
                      showgrid=True, gridcolor="#E5E7EB")

fig2.update_layout(
    title={"text": "Surekli Degiskenler: Boxplot Analizi",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", plot_bgcolor="#FBFBF8",
    height=500, font={"family": "Arial", "size": 12, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60)
)
save_figure(fig2, "phase2_boxplots_continuous")

# ─────────────────────────────────────────────
# 2. BALANCE: SIFIR DEGERLERI
# ─────────────────────────────────────────────
zero_balance = (df["Balance"] == 0).sum()
zero_pct     = zero_balance / len(df) * 100
print(f"\nBalance=0 olan musteri sayisi: {zero_balance} ({zero_pct:.1f}%)")
add_rec(
    issue="Balance=0 yogunlugu",
    evidence=f"Balance degiskeni musterilerin %{zero_pct:.1f}'inde sifir.",
    recommendation="Sifir-bakiye bayrak degiskeni (has_zero_balance) olusturulmasi onerilir.",
    priority="Orta"
)

# ─────────────────────────────────────────────
# 3. KATEGORIK DEGISKENLER
# ─────────────────────────────────────────────
print("\n--- Kategorik Degiskenler ---")

cat_records = []
for col in CAT_COLS:
    vc = df[col].value_counts()
    vr = df[col].value_counts(normalize=True) * 100
    dominant = vr.max()
    print(f"\n  {col}:")
    for cat, cnt in vc.items():
        print(f"    {cat:15s}: {cnt:5d}  (%{vr[cat]:.1f})")
    cat_records.append({"Degisken": col, "Uniq": df[col].nunique(), "Baskın_%": round(dominant, 2)})

# ─── Geography bar ───
geo_vc = df["Geography"].value_counts().reset_index()
geo_vc.columns = ["Ulke", "Adet"]
geo_vc["Oran_%"] = (geo_vc["Adet"] / len(df) * 100).round(1)

fig3 = px.bar(geo_vc, x="Ulke", y="Adet",
              text=geo_vc["Oran_%"].apply(lambda x: f"%{x}"),
              color="Ulke",
              color_discrete_sequence=PROFESSIONAL_PALETTE)
fig3 = apply_premium_layout(fig3, "Cografya Dagilimi (Geography)", height=450)
fig3.update_traces(textposition="outside")
fig3.update_layout(xaxis_title="Ulke", yaxis_title="Musteri Sayisi", showlegend=False)
save_figure(fig3, "phase2_bar_geography")

# ─── Gender pie ───
gen_vc = df["Gender"].value_counts().reset_index()
gen_vc.columns = ["Cinsiyet", "Adet"]
fig4 = px.pie(gen_vc, names="Cinsiyet", values="Adet",
              color_discrete_sequence=[PROFESSIONAL_PALETTE[0], PROFESSIONAL_PALETTE[1]],
              hole=0.35)
fig4 = apply_premium_layout(fig4, "Cinsiyet Dagilimi (Gender)", height=450)
fig4.update_traces(textinfo="label+percent", textfont_size=14)
save_figure(fig4, "phase2_pie_gender")

# ─────────────────────────────────────────────
# 4. IKILI (BINARY) DEGISKENLER
# ─────────────────────────────────────────────
print("\n--- Ikili Degiskenler ---")
fig5 = make_subplots(rows=1, cols=3,
    subplot_titles=["<b>NumOfProducts</b>", "<b>HasCrCard</b>", "<b>IsActiveMember</b>"],
    horizontal_spacing=0.12)

disc_cols_all = ["NumOfProducts", "HasCrCard", "IsActiveMember"]
for idx, col in enumerate(disc_cols_all, 1):
    vc = df[col].value_counts().sort_index()
    pct = (vc / len(df) * 100).round(1)
    fig5.add_trace(
        go.Bar(x=vc.index.astype(str), y=vc.values,
               text=[f"%{p}" for p in pct.values],
               textposition="outside",
               marker_color=PROFESSIONAL_PALETTE[idx-1],
               name=col, showlegend=False),
        row=1, col=idx
    )
    fig5.update_xaxes(title_text=col, row=1, col=idx)
    fig5.update_yaxes(title_text="Adet", row=1, col=idx,
                      showgrid=True, gridcolor="#E5E7EB")

fig5.update_layout(
    title={"text": "Diskret / Ikili Degiskenler: Frekans Dagilimi",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", height=480,
    font={"family": "Arial", "size": 13, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60)
)
save_figure(fig5, "phase2_bar_discrete_binary")

# ─────────────────────────────────────────────
# 5. HEDEF DEGİSKEN: EXITED
# ─────────────────────────────────────────────
target_vc = df["Exited"].value_counts().reset_index()
target_vc.columns = ["Sinif", "Adet"]
target_vc["Etiket"] = target_vc["Sinif"].map({0: "Kaldi (0)", 1: "Ayrildi (1)"})
target_vc["Oran_%"] = (target_vc["Adet"] / len(df) * 100).round(1)

fig6 = px.bar(target_vc, x="Etiket", y="Adet",
              text=target_vc["Oran_%"].apply(lambda x: f"%{x}"),
              color="Etiket",
              color_discrete_sequence=["#2E86AB", "#C73E1D"])
fig6 = apply_premium_layout(fig6, "Hedef Degisken Dagilimi: Exited (Churn)", height=480)
fig6.update_traces(textposition="outside", textfont_size=15)
fig6.update_layout(
    xaxis_title="Musteri Durumu", yaxis_title="Musteri Sayisi", showlegend=False,
    annotations=[dict(
        x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False,
        text="Sinif dengesizligi: Churn orani %20.3 — SMOTE / Class Weighting onerilir",
        font=dict(size=13, color="#C73E1D")
    )]
)
save_figure(fig6, "phase2_target_exited_distribution")

dominant_ratio = target_vc["Oran_%"].max()
if dominant_ratio > 70:
    add_rec(
        issue="Hedef degiskende dengesiz dagilim",
        evidence=f"Baskin sinif orani %{dominant_ratio:.1f} (Kaldi=0).",
        recommendation="SMOTE, class weighting veya stratified split onerilir.",
        priority="Yuksek"
    )

# ─────────────────────────────────────────────
# KAYIT
# ─────────────────────────────────────────────
skew_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase2_univariate_stats.csv", index=False)

recs_df = pd.DataFrame(data_prep_recommendations)
recs_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase2_data_prep_recommendations.csv", index=False)

print(f"\nPhase 2 tamamlandi. {len(data_prep_recommendations)} Data Prep onerisi kaydedildi.")
print("\nONERILER:")
for r in data_prep_recommendations:
    print(f"  [{r['Oncelik']}] {r['Sorun']}: {r['Kanit'][:80]}")
