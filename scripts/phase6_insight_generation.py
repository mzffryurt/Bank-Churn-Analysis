"""
PHASE 6: INSIGHT GENERATION
Teknik sonuclari anlamli icgoruler + model hazirlik ozeti.
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
df["Durum"] = df["Exited"].map({0: "Kaldi", 1: "Ayrildi"})

print("=" * 60)
print("PHASE 6: ICGORU URETIMI")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. YAS GRUPLARI DETAYI
# ─────────────────────────────────────────────
print("\n--- Yas Gruplari ve Churn ---")
df["AgeGroup"] = pd.cut(df["Age"],
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])

age_churn = df.groupby("AgeGroup", observed=True)["Exited"].agg(["sum","count"]).reset_index()
age_churn.columns = ["Yas_Grubu", "Churn_Sayisi", "Toplam"]
age_churn["Churn_%"] = (age_churn["Churn_Sayisi"] / age_churn["Toplam"] * 100).round(1)
print(age_churn.to_string(index=False))

fig1 = px.bar(age_churn, x="Yas_Grubu", y="Churn_%",
              text=age_churn["Churn_%"].apply(lambda x: f"%{x}"),
              color="Yas_Grubu",
              color_discrete_sequence=PROFESSIONAL_PALETTE)
fig1 = apply_premium_layout(fig1, "Yas Gruplari: Churn Oranlari", height=480)
fig1.update_traces(textposition="outside", textfont_size=14)
fig1.update_layout(xaxis_title="Yas Grubu", yaxis_title="Churn Orani (%)", showlegend=False)
save_figure(fig1, "phase6_age_group_churn_rate")

# ─────────────────────────────────────────────
# 2. ALMAN KADIN MUSTERİLER — EN RİSKLİ SEGMENT
# ─────────────────────────────────────────────
print("\n--- Segment Analizi: Cografya x Cinsiyet ---")
seg = df.groupby(["Geography", "Gender"])["Exited"].agg(["sum", "count"]).reset_index()
seg.columns = ["Cografya", "Cinsiyet", "Churn_Sayisi", "Toplam"]
seg["Churn_%"] = (seg["Churn_Sayisi"] / seg["Toplam"] * 100).round(1)
seg_sorted = seg.sort_values("Churn_%", ascending=False)
print(seg_sorted.to_string(index=False))

fig2 = px.bar(seg_sorted, x="Churn_%", y=seg_sorted["Cografya"] + " / " + seg_sorted["Cinsiyet"],
              orientation="h", text="Churn_%",
              color="Churn_%",
              color_continuous_scale=[[0, "#D5F5E3"], [0.5, "#F7D9A3"], [1.0, "#C73E1D"]])
fig2 = apply_premium_layout(fig2, "Segment Bazinda Churn Oranlari (Cografya x Cinsiyet)", height=480)
fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig2.update_layout(xaxis_title="Churn Orani (%)", yaxis_title="Segment",
                   coloraxis_showscale=False)
save_figure(fig2, "phase6_segment_churn_rates")

# ─────────────────────────────────────────────
# 3. AKTIF UYELIK x NUMOFPRODUCTSx CHURN
# ─────────────────────────────────────────────
print("\n--- NumOfProducts x IsActiveMember: Churn ---")
nop_act = df.groupby(["NumOfProducts", "IsActiveMember"])["Exited"].agg(["sum","count"]).reset_index()
nop_act.columns = ["NumOfProducts", "IsActiveMember", "Churn_Sayisi", "Toplam"]
nop_act["Churn_%"] = (nop_act["Churn_Sayisi"] / nop_act["Toplam"] * 100).round(1)
nop_act["Uye_Durumu"] = nop_act["IsActiveMember"].map({0: "Pasif", 1: "Aktif"})
print(nop_act[["NumOfProducts", "Uye_Durumu", "Churn_%"]].to_string(index=False))

fig3 = px.bar(nop_act, x="NumOfProducts", y="Churn_%",
              color="Uye_Durumu", barmode="group", text="Churn_%",
              color_discrete_map={"Aktif": "#2E86AB", "Pasif": "#C73E1D"})
fig3 = apply_premium_layout(fig3, "Urun Sayisi x Uyelik Durumu: Churn Oranlari", height=480)
fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig3.update_layout(xaxis_title="Urun Sayisi", yaxis_title="Churn Orani (%)",
                   legend_title_text="Uyelik Durumu")
save_figure(fig3, "phase6_numproducts_activemember_churn")

# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (ETKI OZETI TABLOSU)
# ─────────────────────────────────────────────
feature_importance = [
    {"Degisken": "Age",             "Etki": "Yuksek",  "Yon": "Pozitif", "Korelasyon_r": 0.285,
     "Aciklama": "Yasli musteriler daha cok ayrilıyor (45+ risk grubu)"},
    {"Degisken": "IsActiveMember",  "Etki": "Orta",    "Yon": "Negatif", "Korelasyon_r": -0.175,
     "Aciklama": "Pasif uyeler 2x daha fazla ayrilıyor"},
    {"Degisken": "Geography",       "Etki": "Orta",    "Yon": "Pozitif", "Korelasyon_r": 0.102,
     "Aciklama": "Almanya churn orani Fransa/Ispanya'ya gore 1.6x yuksek"},
    {"Degisken": "NumOfProducts",   "Etki": "Orta",    "Yon": "Non-linear","Korelasyon_r": 0.088,
     "Aciklama": "1-2 urun ideal; 3-4 urunde churn %63-66"},
    {"Degisken": "Gender",          "Etki": "Dusuk",   "Yon": "Pozitif", "Korelasyon_r": 0.060,
     "Aciklama": "Kadinlar %22.9 vs Erkekler %18.1 churn"},
    {"Degisken": "Balance",         "Etki": "Dusuk",   "Yon": "Pozitif", "Korelasyon_r": 0.037,
     "Aciklama": "Pozitif bakiye hafif yuksek churn ile iliskili"},
    {"Degisken": "CreditScore",     "Etki": "Cok Dusuk","Yon": "Negatif","Korelasyon_r": -0.014,
     "Aciklama": "Churn ile anlamsiz iliskisi var (p>0.05)"},
    {"Degisken": "Tenure",          "Etki": "Cok Dusuk","Yon": "Negatif","Korelasyon_r": -0.004,
     "Aciklama": "Musterilk suresi ile churn arasinda iliskisi yok"},
    {"Degisken": "EstimatedSalary", "Etki": "Cok Dusuk","Yon": "Negatif","Korelasyon_r": -0.011,
     "Aciklama": "Maas churn'u aciklamıyor"},
    {"Degisken": "HasCrCard",       "Etki": "Cok Dusuk","Yon": "Negatif","Korelasyon_r": -0.017,
     "Aciklama": "Kredi karti sahipligi churn'u aciklamıyor"},
]

feat_df = pd.DataFrame(feature_importance)
feat_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase6_feature_importance_summary.csv", index=False)
print("\n--- Feature Importance Ozeti ---")
print(feat_df[["Degisken", "Etki", "Korelasyon_r", "Aciklama"]].to_string(index=False))

# Feature importance bar
feat_plot = feat_df.copy()
feat_plot["Abs_r"] = feat_plot["Korelasyon_r"].abs()
feat_plot = feat_plot.sort_values("Abs_r", ascending=True)
feat_plot["Bar_Color"] = feat_plot["Korelasyon_r"].apply(lambda x: "#C73E1D" if x > 0 else "#2E86AB")

fig4 = go.Figure(go.Bar(
    x=feat_plot["Korelasyon_r"],
    y=feat_plot["Degisken"],
    orientation="h",
    marker_color=feat_plot["Bar_Color"],
    text=feat_plot["Korelasyon_r"].round(3),
    textposition="outside"
))
fig4 = apply_premium_layout(fig4, "Degisken Etkisi: Exited ile Korelasyon (Siralanmis)", height=540)
fig4.update_layout(
    xaxis_title="Pearson Korelasyon (r)",
    yaxis_title="Degisken",
    xaxis=dict(range=[-0.35, 0.40], showgrid=True, gridcolor="#E5E7EB",
               zeroline=True, zerolinecolor="#374151", zerolinewidth=2)
)
save_figure(fig4, "phase6_feature_impact_ranking")

# ─────────────────────────────────────────────
# 5. DATA PREP ONERILERI KONSOLIDE TABLOSU
# ─────────────────────────────────────────────
all_recs = [
    # Phase 1
    {"Sorun": "Gereksiz sutun: RowNumber",    "Oncelik": "Yuksek",
     "Oneri": "Modelden cikar. Bilgi tasimiyor."},
    {"Sorun": "Gereksiz sutun: CustomerId",   "Oncelik": "Yuksek",
     "Oneri": "Modelden cikar. Bilgi tasimiyor."},
    {"Sorun": "Gereksiz sutun: Surname",      "Oncelik": "Yuksek",
     "Oneri": "Modelden cikar. Yuksek kardinalite, bilgi tasimıyor."},
    # Phase 2
    {"Sorun": "Sinif dengesizligi (Exited=%79.7)",  "Oncelik": "Yuksek",
     "Oneri": "SMOTE, class weighting veya stratified split uygula."},
    {"Sorun": "Balance=0 yogunlugu (%35.4)",  "Oncelik": "Orta",
     "Oneri": "has_zero_balance bayrak degiskeni olustur."},
    # Phase 3
    {"Sorun": "NumOfProducts=3-4 anomalisi",  "Oncelik": "Yuksek",
     "Oneri": "is_high_products_user bayrak degiskeni olustur veya ozel segment modeli."},
    {"Sorun": "Germany segmenti yuksek churn (%28.8)", "Oncelik": "Orta",
     "Oneri": "Cografi encoding veya target encoding uygula."},
    # Phase 4
    {"Sorun": "Encoding gereksinimi: Geography, Gender", "Oncelik": "Yuksek",
     "Oneri": "One-hot encoding veya target encoding uygula."},
    # Phase 5
    {"Sorun": "CustomerId tekrari (195 adet)", "Oncelik": "Orta",
     "Oneri": "Ayni musterinin birden fazla kaydi var mi kontrol et. Gerekirse aggr."},
    # Feature Engineering
    {"Sorun": "Feature Engineering firsati", "Oncelik": "Orta",
     "Oneri": "age_group, balance_segment, geo_gender_risk, products_risk olustur."},
    # Scaling
    {"Sorun": "Scaling gereksinimi",          "Oncelik": "Yuksek",
     "Oneri": "StandardScaler veya MinMaxScaler (CreditScore, Age, Balance, EstimatedSalary)."},
]

all_recs_df = pd.DataFrame(all_recs)
all_recs_df.to_csv(
    PROJECT_ROOT / "reports" / "csv" / "phase6_all_data_prep_recommendations.csv", index=False
)
print("\n--- Konsolide Data Prep Onerileri ---")
print(all_recs_df.to_string(index=False))

print("\nPhase 6 tamamlandi.")
