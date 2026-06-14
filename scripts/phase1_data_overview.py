"""
PHASE 1: DATA OVERVIEW
Veri setinin temel yapısını anlamak için genel bakış analizi.
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"C:\Users\sence\OneDrive\Desktop\churn-analysis")

# Klasörlerin varlığını garantile
(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "reports" / "csv").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "reports" / "markdown").mkdir(parents=True, exist_ok=True)

# Profesyonel renk paleti
PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
    "#BC4B51", "#8E7DBE", "#F77F00", "#06A77D", "#D4A574"
]

data_prep_recommendations = []

def add_data_prep_recommendation(issue, evidence, recommendation, priority="Orta"):
    data_prep_recommendations.append({
        "Sorun": issue,
        "Kanit": evidence,
        "Oneri": recommendation,
        "Oncelik": priority
    })

def apply_premium_layout(fig, title):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 22, "family": "Arial Black", "color": "#1F2937"}
        },
        template="plotly_white",
        paper_bgcolor="#FBFBF8",
        plot_bgcolor="#FBFBF8",
        font={"family": "Arial", "size": 13, "color": "#374151"},
        margin=dict(l=60, r=40, t=80, b=60),
        legend_title_text="Kategori",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig

def save_figure(fig, file_base):
    html_path = PROJECT_ROOT / "figures" / f"{file_base}.html"
    png_path = PROJECT_ROOT / "figures" / f"{file_base}.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path))
        print(f"  PNG kaydedildi: figures/{file_base}.png")
    except Exception as e:
        print(f"  PNG kaydı yapılamadı (kaleido eksik olabilir): {e}")
    print(f"  HTML kaydedildi: figures/{file_base}.html")
    return html_path, png_path

# ─────────────────────────────────────────────
# VERİYİ YÜKLE
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: VERİ GENEL BAKIŞ")
print("=" * 60)

df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "churn.csv")

print(f"\nSatır sayısı     : {len(df):,}")
print(f"Sütun sayısı     : {len(df.columns)}")
print(f"Toplam hücre     : {df.size:,}")
print(f"Bellek kullanımı : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"\nSütunlar: {list(df.columns)}")

# ─────────────────────────────────────────────
# VERİ TİPLERİ
# ─────────────────────────────────────────────
print("\n--- Veri Tipleri ---")
dtype_df = pd.DataFrame({
    "Sutun": df.columns,
    "Tip": df.dtypes.values.astype(str),
    "Bos_Deger": df.isnull().sum().values,
    "Bos_Oran_%": (df.isnull().sum().values / len(df) * 100).round(2),
    "Essiz_Deger": df.nunique().values
})
print(dtype_df.to_string(index=False))

# ─────────────────────────────────────────────
# SAYISAL ÖZETİ
# ─────────────────────────────────────────────
print("\n--- Sayısal Değişkenler Özet İstatistikleri ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(df[numeric_cols].describe().round(2).to_string())

print(f"\nSayısal değişkenler: {numeric_cols}")
print(f"Kategorik değişkenler: {cat_cols}")

# ─────────────────────────────────────────────
# EKSİK VERİ
# ─────────────────────────────────────────────
total_missing = df.isnull().sum().sum()
print(f"\nToplam eksik değer: {total_missing}")
if total_missing == 0:
    print("Veri setinde eksik değer bulunmamaktadır.")

# ─────────────────────────────────────────────
# MÜKERRER KAYIT
# ─────────────────────────────────────────────
duplicates = df.duplicated().sum()
print(f"Mükerrer kayıt sayısı: {duplicates}")

# ─────────────────────────────────────────────
# ID SÜTUNLARI TESPİTİ
# ─────────────────────────────────────────────
id_cols = ["RowNumber", "CustomerId", "Surname"]
print(f"\nID / Gereksiz sütunlar (modelden çıkarılmalı): {id_cols}")
for col in id_cols:
    print(f"  {col}: {df[col].nunique()} eşsiz değer")
    add_data_prep_recommendation(
        issue=f"Gereksiz/kimlik sütunu: {col}",
        evidence=f"{col} sütunu modelleme için bilgi taşımaz ({df[col].nunique()} eşsiz değer).",
        recommendation=f"Data Prep Expert {col} sütununu modelden çıkarmalıdır.",
        priority="Yüksek"
    )

# ─────────────────────────────────────────────
# HEDEF DEĞİŞKEN
# ─────────────────────────────────────────────
print("\n--- Hedef Değişken: Exited ---")
target_counts = df["Exited"].value_counts()
target_ratio = df["Exited"].value_counts(normalize=True) * 100
print(f"  0 (Kalmış) : {target_counts[0]:,} ({target_ratio[0]:.1f}%)")
print(f"  1 (Ayrılmış): {target_counts[1]:,} ({target_ratio[1]:.1f}%)")

# ─────────────────────────────────────────────
# VERİ YAPISI GÖRSELİ
# ─────────────────────────────────────────────
dtype_map = {col: str(df[col].dtype) for col in df.columns}
dtype_color = {
    "int64": "#2E86AB",
    "float64": "#06A77D",
    "object": "#F18F01"
}

fig = go.Figure()
for i, col in enumerate(df.columns):
    dtype = dtype_map[col]
    color = dtype_color.get(dtype, "#8E7DBE")
    fig.add_trace(go.Bar(
        x=[col],
        y=[df[col].nunique()],
        name=dtype,
        marker_color=color,
        text=[f"{df[col].nunique()} eşsiz<br>{dtype}"],
        textposition="outside",
        showlegend=(i < 3)
    ))

fig = apply_premium_layout(fig, "Veri Seti Sütun Yapısı: Eşsiz Değer Sayıları ve Tipler")
fig.update_layout(
    xaxis_title="Sütun Adı",
    yaxis_title="Eşsiz Değer Sayısı",
    showlegend=True,
    barmode="group",
    height=500
)
save_figure(fig, "phase1_column_overview")

# CSV kaydet
dtype_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "phase1_data_overview.csv", index=False)

# Önerileri kaydet
recs_df = pd.DataFrame(data_prep_recommendations)
recs_df.to_csv(PROJECT_ROOT / "reports" / "csv" / "data_prep_recommendations.csv", index=False)

print("\nPhase 1 tamamlandı.")
print(f"Data Prep önerileri: {len(data_prep_recommendations)} adet kaydedildi.")
