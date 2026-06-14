"""
PHASE 3: BIVARIATE ANALYSIS
Her ozelligin Exited (hedef) ile iliskisi.
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
COLOR_NO_CHURN  = "#2E86AB"  # Mavi - kaldi
COLOR_CHURN     = "#C73E1D"  # Kirmizi - ayrildi

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
CONT_COLS  = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CAT_COLS   = ["Geography", "Gender"]
DISC_COLS  = ["NumOfProducts", "HasCrCard", "IsActiveMember"]

print("=" * 60)
print("PHASE 3: IKI DEGİSKENLI ANALIZ (vs Exited)")
print("=" * 60)

bivariate_records = []

# ─────────────────────────────────────────────
# 1. SAYISAL vs EXITED — BOXPLOT PANELİ
# ─────────────────────────────────────────────
print("\n--- Surekli Degiskenler vs Exited ---")
df["Durum"] = df["Exited"].map({0: "Kaldi", 1: "Ayrildi"})

fig = make_subplots(rows=2, cols=3,
    subplot_titles=[f"<b>{c}</b>" for c in CONT_COLS],
    horizontal_spacing=0.10, vertical_spacing=0.20)

row_col = [(1,1),(1,2),(1,3),(2,1),(2,2)]
for idx, col in enumerate(CONT_COLS):
    r, c = row_col[idx]
    for sinif, color, name in [(0, COLOR_NO_CHURN, "Kaldi"), (1, COLOR_CHURN, "Ayrildi")]:
        sub = df[df["Exited"] == sinif]
        fig.add_trace(
            go.Box(y=sub[col], name=name, marker_color=color,
                   line_color=color, opacity=0.8,
                   showlegend=(idx == 0), legendgroup=name),
            row=r, col=c
        )
    fig.update_xaxes(title_text="Musteri Durumu", row=r, col=c)
    fig.update_yaxes(title_text=col, row=r, col=c,
                     showgrid=True, gridcolor="#E5E7EB")

    # Mann-Whitney U istatistigi
    stayed  = df[df["Exited"] == 0][col].dropna()
    churned = df[df["Exited"] == 1][col].dropna()
    u_stat, p_val = stats.mannwhitneyu(stayed, churned, alternative="two-sided")
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

    mean_stayed  = stayed.mean()
    mean_churned = churned.mean()
    diff_pct = (mean_churned - mean_stayed) / mean_stayed * 100
    print(f"  {col:20s} | Kaldi={mean_stayed:.1f} | Ayrildi={mean_churned:.1f} | fark=%{diff_pct:+.1f} | p={p_val:.4f} {sig}")
    bivariate_records.append({
        "Degisken": col, "Kaldi_Ort": round(mean_stayed, 2),
        "Ayrildi_Ort": round(mean_churned, 2), "Fark_%": round(diff_pct, 2),
        "MW_p": round(p_val, 6), "Anlamlilik": sig
    })

fig.update_layout(
    title={"text": "Surekli Degiskenler: Churn (Ayrildi) vs. Kaldi — Boxplot",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", height=640,
    legend=dict(x=0.82, y=0.15, bgcolor="white", bordercolor="#E5E7EB"),
    font={"family": "Arial", "size": 13, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60)
)
save_figure(fig, "phase3_boxplot_numeric_vs_exited")

# ─────────────────────────────────────────────
# 2. YAS DAGILIMI: CHURN vs KALDI (HISTOGRAM)
# ─────────────────────────────────────────────
fig2 = go.Figure()
for sinif, color, name in [(0, COLOR_NO_CHURN, "Kaldi"), (1, COLOR_CHURN, "Ayrildi")]:
    sub = df[df["Exited"] == sinif]
    fig2.add_trace(go.Histogram(
        x=sub["Age"], nbinsx=40, name=name,
        marker_color=color, opacity=0.75,
        histnorm="probability density"
    ))
fig2 = apply_premium_layout(fig2, "Yas Dagilimi: Churn vs Kaldi", height=480)
fig2.update_layout(barmode="overlay", xaxis_title="Yas", yaxis_title="Yoğunluk",
                   legend=dict(x=0.82, y=0.95))
save_figure(fig2, "phase3_histogram_age_vs_exited")

# ─────────────────────────────────────────────
# 3. BALANCE: CHURN ORANI (SIFIR vs POZITIF)
# ─────────────────────────────────────────────
df["Balance_Grubu"] = np.where(df["Balance"] == 0, "Sifir Bakiye", "Pozitif Bakiye")
bal_churn = df.groupby("Balance_Grubu")["Exited"].mean().reset_index()
bal_churn.columns = ["Bakiye_Grubu", "Churn_Orani"]
bal_churn["Churn_%"] = (bal_churn["Churn_Orani"] * 100).round(1)

fig3 = px.bar(bal_churn, x="Bakiye_Grubu", y="Churn_%",
              text=bal_churn["Churn_%"].apply(lambda x: f"%{x}"),
              color="Bakiye_Grubu",
              color_discrete_sequence=[PROFESSIONAL_PALETTE[2], PROFESSIONAL_PALETTE[0]])
fig3 = apply_premium_layout(fig3, "Balance Grubu vs Churn Orani", height=450)
fig3.update_traces(textposition="outside", textfont_size=15)
fig3.update_layout(xaxis_title="Bakiye Grubu", yaxis_title="Churn Orani (%)", showlegend=False)
save_figure(fig3, "phase3_balance_group_churn_rate")
print(f"\n  Balance Grubu Churn Oranlari:\n{bal_churn.to_string(index=False)}")

# ─────────────────────────────────────────────
# 4. KATEGORİK vs EXITED
# ─────────────────────────────────────────────
print("\n--- Kategorik Degiskenler vs Exited ---")

for col in CAT_COLS:
    ct = pd.crosstab(df[col], df["Exited"], normalize="index") * 100
    churn_rate = ct[1].sort_values(ascending=False)
    print(f"\n  {col} - Churn Oranlari:")
    for cat, rate in churn_rate.items():
        print(f"    {cat:15s}: %{rate:.1f}")

# Grouped bar: Geography vs Churn
geo_churn = df.groupby("Geography")["Exited"].agg(["sum", "count"])
geo_churn["Churn_Orani_%"] = (geo_churn["sum"] / geo_churn["count"] * 100).round(1)
geo_churn = geo_churn.reset_index()
geo_churn.columns = ["Ulke", "Churn_Sayisi", "Toplam", "Churn_%"]

fig4 = make_subplots(rows=1, cols=2,
    subplot_titles=["<b>Ulke Bazinda Churn Orani (%)</b>", "<b>Cinsiyet Bazinda Churn Orani (%)</b>"],
    horizontal_spacing=0.15)

fig4.add_trace(
    go.Bar(x=geo_churn["Ulke"], y=geo_churn["Churn_%"],
           text=[f"%{v}" for v in geo_churn["Churn_%"]],
           textposition="outside",
           marker_color=[PROFESSIONAL_PALETTE[0], PROFESSIONAL_PALETTE[1], PROFESSIONAL_PALETTE[2]],
           showlegend=False),
    row=1, col=1
)

gen_churn = df.groupby("Gender")["Exited"].agg(["sum", "count"])
gen_churn["Churn_%"] = (gen_churn["sum"] / gen_churn["count"] * 100).round(1)
gen_churn = gen_churn.reset_index()
gen_churn.columns = ["Cinsiyet", "Churn_Sayisi", "Toplam", "Churn_%"]

fig4.add_trace(
    go.Bar(x=gen_churn["Cinsiyet"], y=gen_churn["Churn_%"],
           text=[f"%{v}" for v in gen_churn["Churn_%"]],
           textposition="outside",
           marker_color=[PROFESSIONAL_PALETTE[3], PROFESSIONAL_PALETTE[4]],
           showlegend=False),
    row=1, col=2
)
fig4.update_layout(
    title={"text": "Kategorik Degiskenler: Churn Oranlari",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", height=500,
    font={"family": "Arial", "size": 13, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60)
)
fig4.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
fig4.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False, title_text="Churn Orani (%)")
save_figure(fig4, "phase3_categorical_churn_rates")

# ─────────────────────────────────────────────
# 5. DİSKRET DEGİSKENLER vs EXITED
# ─────────────────────────────────────────────
print("\n--- Diskret Degiskenler vs Exited ---")

fig5 = make_subplots(rows=1, cols=3,
    subplot_titles=[f"<b>{c}</b>" for c in DISC_COLS],
    horizontal_spacing=0.12)

for idx, col in enumerate(DISC_COLS, 1):
    grp = df.groupby(col)["Exited"].agg(["sum", "count"])
    grp["Churn_%"] = (grp["sum"] / grp["count"] * 100).round(1)
    grp = grp.reset_index()
    print(f"\n  {col}:")
    print(grp[[col, "Churn_%"]].to_string(index=False))

    fig5.add_trace(
        go.Bar(x=grp[col].astype(str), y=grp["Churn_%"],
               text=[f"%{v}" for v in grp["Churn_%"]],
               textposition="outside",
               marker_color=PROFESSIONAL_PALETTE[idx-1],
               name=col, showlegend=False),
        row=1, col=idx
    )
    fig5.update_xaxes(title_text=col, row=1, col=idx)
    fig5.update_yaxes(title_text="Churn (%)", row=1, col=idx,
                      showgrid=True, gridcolor="#E5E7EB")

fig5.update_layout(
    title={"text": "Diskret / Ikili Degiskenler: Churn Oranlari",
           "x": 0.03, "xanchor": "left",
           "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}},
    template="plotly_white", paper_bgcolor="#FBFBF8", height=500,
    font={"family": "Arial", "size": 13, "color": "#374151"},
    margin=dict(l=60, r=40, t=90, b=60)
)
save_figure(fig5, "phase3_discrete_churn_rates")

# NumOfProducts=4 anomalisi
nop4 = df[df["NumOfProducts"] == 4]["Exited"].mean() * 100
print(f"\n  NumOfProducts=4 churn orani: %{nop4:.1f}")
if nop4 > 80:
    add_rec(
        issue="NumOfProducts=4 anomalisi",
        evidence=f"4 urun sahibi musterilerin churn orani %{nop4:.1f} — asiri yuksek.",
        recommendation="Bu grup icin ozel kural/flag olusturulabilir veya ayri segment modeli degerlendirilmeli.",
        priority="Yuksek"
    )

# ─────────────────────────────────────────────
# 6. YAS GRUPLARİ vs CHURN (VIOLIN)
# ─────────────────────────────────────────────
fig6 = go.Figure()
fig6.add_trace(go.Violin(
    y=df[df["Exited"] == 0]["Age"], name="Kaldi",
    line_color=COLOR_NO_CHURN, fillcolor=COLOR_NO_CHURN, opacity=0.7,
    box_visible=True, meanline_visible=True
))
fig6.add_trace(go.Violin(
    y=df[df["Exited"] == 1]["Age"], name="Ayrildi",
    line_color=COLOR_CHURN, fillcolor=COLOR_CHURN, opacity=0.7,
    box_visible=True, meanline_visible=True
))
fig6 = apply_premium_layout(fig6, "Yas Dagilimi: Kaldi vs Ayrildi (Violin)", height=500)
fig6.update_layout(xaxis_title="Musteri Durumu", yaxis_title="Yas",
                   violinmode="group",
                   legend=dict(x=0.02, y=0.98))
save_figure(fig6, "phase3_violin_age_vs_exited")

# ─────────────────────────────────────────────
# 7. STACKED BAR: GEOGRAPHY x GENDER x CHURN
# ─────────────────────────────────────────────
geo_gen = df.groupby(["Geography", "Gender"])["Exited"].agg(["sum","count"]).reset_index()
geo_gen["Churn_%"] = (geo_gen["sum"] / geo_gen["count"] * 100).round(1)

fig7 = px.bar(geo_gen, x="Geography", y="Churn_%", color="Gender",
              barmode="group", text="Churn_%",
              color_discrete_sequence=[PROFESSIONAL_PALETTE[0], PROFESSIONAL_PALETTE[1]])
fig7 = apply_premium_layout(fig7, "Ulke x Cinsiyet: Churn Oranlari", height=480)
fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig7.update_layout(xaxis_title="Ulke", yaxis_title="Churn Orani (%)",
                   legend_title_text="Cinsiyet")
save_figure(fig7, "phase3_geo_gender_churn")

# ─────────────────────────────────────────────
# KAYIT
# ─────────────────────────────────────────────
biv_df = pd.DataFrame(bivariate_records)
biv_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase3_bivariate_stats.csv", index=False)

recs_df = pd.DataFrame(data_prep_recommendations)
recs_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase3_data_prep_recommendations.csv", index=False)

print(f"\nPhase 3 tamamlandi. {len(data_prep_recommendations)} Data Prep onerisi kaydedildi.")
print("\nIstatistiksel Anlamlilik Ozeti:")
print(biv_df.to_string(index=False))
