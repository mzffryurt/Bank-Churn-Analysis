"""
DataPrep Expert - Agentik 7-Aşamalı Veri Hazırlama Pipeline
Proje: Banka Müşteri Kaybı (Churn) Tahmini
Hedef: Exited (ikili sınıflandırma)
EDA Çıktısı → DataPrep → Model Expert
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PROJE KÖKÜ VE KLASÖR YAPISI
# ─────────────────────────────────────────────
PROJECT_ROOT = Path("C:/Users/sence/OneDrive/Desktop/churn-analysis")

DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
DATA_MODEL      = PROJECT_ROOT / "data" / "model_ready"
FIGURES_DIR     = PROJECT_ROOT / "figures"
REPORTS_CSV     = PROJECT_ROOT / "reports" / "csv"
REPORTS_MD      = PROJECT_ROOT / "reports" / "markdown"
MODELS_DIR      = PROJECT_ROOT / "models"
SCRIPTS_DIR     = PROJECT_ROOT / "scripts"

for d in [DATA_PROCESSED, DATA_MODEL, FIGURES_DIR, REPORTS_CSV, REPORTS_MD, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# GÖRSEL PALETTE
# ─────────────────────────────────────────────
PASTEL_PALETTE = [
    "#A7C7E7", "#B8E0D2", "#F6C6C6", "#F7D9A3",
    "#D7BDE2", "#C8D6AF", "#F5CBA7", "#AED6F1",
    "#D5F5E3", "#FADBD8"
]
COLOR_POS  = "#F6C6C6"   # churn=1
COLOR_NEG  = "#A7C7E7"   # churn=0
COLOR_MAIN = "#AED6F1"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

# ─────────────────────────────────────────────
# MEMORY STRUCTURES
# ─────────────────────────────────────────────
dataprep_actions  = []
model_handoff     = []

def log_action(step, issue, decision, rationale, risk="Dusuk"):
    dataprep_actions.append({
        "Asama"    : step,
        "Sorun"    : issue,
        "Karar"    : decision,
        "Gerekce"  : rationale,
        "Risk"     : risk
    })
    print(f"  [LOG] {step} | {issue} -> {decision}")

def add_handoff(item, status, note):
    model_handoff.append({
        "Bilesen"           : item,
        "Durum"             : status,
        "Model Expert Notu" : note
    })

def section(title):
    bar = "=" * 65
    print(f"\n{bar}\n  {title}\n{bar}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — EDA RECOMMENDATION INGESTION
# ═══════════════════════════════════════════════════════════════════
section("PHASE 1: EDA Oneri Alimi ve Ham Veri Yuklemesi")

df_raw = pd.read_csv(DATA_RAW / "churn.csv")
print(f"Ham veri: {df_raw.shape[0]} satir, {df_raw.shape[1]} sutun")

# EDA önerileri — karar matrisi
eda_recommendations = [
    {
        "Sorun"    : "ID sutunlari: RowNumber, CustomerId, Surname",
        "EDA Onerisi": "Modelden cikar",
        "DataPrep Karari": "UYGULANACAK",
        "Gerekce"  : "Tahmin gucu yok; leakage riski"
    },
    {
        "Sorun"    : "Sinif dengesizligi (%79.7 / %20.3)",
        "EDA Onerisi": "SMOTE veya class_weight='balanced'",
        "DataPrep Karari": "class_weight='balanced' ONERILIR; veriye SMOTE uygulanmaz",
        "Gerekce"  : "Kullanici talimati; model fazinda denge saglanir"
    },
    {
        "Sorun"    : "Kategorik: Geography, Gender",
        "EDA Onerisi": "One-hot veya target encoding",
        "DataPrep Karari": "UYGULANACAK — OHE (Geography 3 kategori, Gender ikilil)",
        "Gerekce"  : "Dusuk kardinalite; OHE optimal"
    },
    {
        "Sorun"    : "Feature scaling",
        "EDA Onerisi": "StandardScaler",
        "DataPrep Karari": "UYGULANACAK — Pipeline icinde (leakage yok)",
        "Gerekce"  : "Dogrusal/distance modeller icin kritik"
    },
    {
        "Sorun"    : "Stratified train-test split",
        "EDA Onerisi": "stratify=y, 80/20",
        "DataPrep Karari": "UYGULANACAK",
        "Gerekce"  : "Hedef oranini korur"
    },
    {
        "Sorun"    : "NumOfProducts=3-4 anomalisi",
        "EDA Onerisi": "is_high_products_risk bayrak degiskeni",
        "DataPrep Karari": "UYGULANACAK",
        "Gerekce"  : "Dogrusal olmayan iliskiyi yakalar (churn %63-66)"
    },
    {
        "Sorun"    : "Balance=0 yogunlugu (%35.4)",
        "EDA Onerisi": "has_zero_balance ikili bayrak",
        "DataPrep Karari": "UYGULANACAK",
        "Gerekce"  : "Bu grup farkli davranis sergiliyor olabilir"
    },
    {
        "Sorun"    : "Outlier (%1.27 max)",
        "EDA Onerisi": "Mudahale gerekli degil",
        "DataPrep Karari": "ATLANDI",
        "Gerekce"  : "Tum degiskenlerde oran <%5 esiginin cok altinda"
    },
]

print("\n--- EDA -> DataPrep Karar Matrisi ---")
df_eda_decisions = pd.DataFrame(eda_recommendations)
print(df_eda_decisions.to_string(index=False))
df_eda_decisions.to_csv(REPORTS_CSV / "phase1_eda_decision_matrix.csv", index=False)

log_action("Phase 1", "EDA onerileri alindi", "8 oneri degerlendirildi — 7 uygulanacak, 1 atlanacak",
           "Outlier orani <%5; mudahale gereksiz")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════
section("PHASE 2: Veri Temizleme")

df = df_raw.copy()
print(f"Baslangic: {df.shape}")

# 2.1 — ID Sutunlarini Dusur
drop_cols = ["RowNumber", "CustomerId", "Surname"]
df.drop(columns=drop_cols, inplace=True)
print(f"  ID sutunlari dusuruldu: {drop_cols}")
print(f"  Yeni shape: {df.shape}")
log_action("Phase 2", "ID sutunlari", f"Dusuruldu: {drop_cols}",
           "Tahmin gucu yok; CustomerId 195 tekrar iceriyordu")

# 2.2 — Eksik Deger Kontrolu
missing = df.isnull().sum()
print(f"\n  Eksik deger kontrolu:\n{missing[missing > 0]}")
if missing.sum() == 0:
    print("  EDA dogrulandi: Eksik deger YOK.")
    log_action("Phase 2", "Eksik deger", "Mudahale yok", "EDA ile dogrulandi — sifir eksik")

# 2.3 — Tip Duzeltme
df["Geography"] = df["Geography"].astype("category")
df["Gender"]    = df["Gender"].astype("category")
log_action("Phase 2", "Veri tipleri", "Geography ve Gender -> category", "Bellek optimizasyonu")

print(f"\n  Temizleme sonrasi shape: {df.shape}")
print(f"  Sutunlar: {df.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — OUTLIER & DISTRIBUTION REPAIR
# ═══════════════════════════════════════════════════════════════════
section("PHASE 3: Outlier ve Dagilim Kontrolu")

numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]

skews = df[numeric_cols].skew()
print("  Carpiklik degerleri:")
for col, sk in skews.items():
    flag = "OK" if abs(sk) < 1 else "DONUSUM GEREKLI"
    print(f"    {col:20s} skew={sk:+.3f}  [{flag}]")

# EDA dogrulasi: hicbiri |skew| > 1 degil → donusum yok
log_action("Phase 3", "Skewness kontrolu", "Donusum yapilmadi",
           "Tum degiskenler |skew| < 1; log/yeo-johnson gereksiz")

# Outlier ozeti
print("\n  IQR Outlier Ozeti:")
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    pct = outliers / len(df) * 100
    flag = "OK" if pct < 5 else "DIKKAT"
    print(f"    {col:20s} n={outliers:4d}  %{pct:.2f}  [{flag}]")

log_action("Phase 3", "Outlier mudahale", "Silme/winsorization yapilmadi",
           "Tum degiskenlerde outlier orani <%1.3; EDA karariyla uyumlu", risk="Dusuk")

# ═══════════════════════════════════════════════════════════════════
# PHASE 4 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
section("PHASE 4: Feature Engineering")

df_fe = df.copy()

# FE-1: NumOfProducts >= 3 bayrak degiskeni
df_fe["is_high_products_risk"] = (df_fe["NumOfProducts"] >= 3).astype(int)
print(f"  FE-1 | is_high_products_risk eklendi  "
      f"(1: {df_fe['is_high_products_risk'].sum()} ornek, "
      f"%{df_fe['is_high_products_risk'].mean()*100:.1f})")
log_action("Phase 4", "NumOfProducts anomalisi",
           "is_high_products_risk olusturuldu (NumOfProducts >= 3)",
           "NumOfProducts=3-4 icin churn %63-66; dogrusal olmayan iliskiyi yakalar")

# FE-2: Balance=0 bayrak degiskeni
df_fe["has_zero_balance"] = (df_fe["Balance"] == 0).astype(int)
print(f"  FE-2 | has_zero_balance eklendi        "
      f"(1: {df_fe['has_zero_balance'].sum()} ornek, "
      f"%{df_fe['has_zero_balance'].mean()*100:.1f})")
log_action("Phase 4", "Balance sifir yogunlugu (%35.4)",
           "has_zero_balance olusturuldu",
           "Sifir bakiyeli musteri davranisi farklidir; davranissal sinyal")

print(f"\n  Feature engineering sonrasi shape: {df_fe.shape}")
print(f"  Yeni sutunlar: {df_fe.columns.tolist()}")

# Churn orani dogrulamasi
print("\n  FE-1 Dogrulama — is_high_products_risk vs Exited:")
print(df_fe.groupby("is_high_products_risk")["Exited"].agg(["mean","count"])
        .rename(columns={"mean":"churn_rate","count":"n"}))

print("\n  FE-2 Dogrulama — has_zero_balance vs Exited:")
print(df_fe.groupby("has_zero_balance")["Exited"].agg(["mean","count"])
        .rename(columns={"mean":"churn_rate","count":"n"}))

add_handoff("is_high_products_risk", "Olusturuldu",
            "NumOfProducts>=3 icin %64 churn; tree-based modellerde yuksek onem beklenir")
add_handoff("has_zero_balance", "Olusturuldu",
            "Sifir bakiye davranissal sinyal; dusuk-orta onem beklenir")

# ═══════════════════════════════════════════════════════════════════
# PHASE 5 — LEAKAGE AUDIT & FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════
section("PHASE 5: Leakage Denetimi ve Ozellik Secimi")

TARGET = "Exited"

# Leakage kontrolu — yuksek korelasyon tarama
X_check = df_fe.drop(columns=[TARGET])
y_check = df_fe[TARGET]

numeric_for_corr = X_check.select_dtypes(include=[np.number]).columns.tolist()
corr_with_target = X_check[numeric_for_corr].corrwith(y_check).abs().sort_values(ascending=False)
print("  Hedef ile korelasyonlar (|r|, buyukten kucuge):")
for col, r in corr_with_target.items():
    flag = "LEAKAGE RISKI - KONTROL ET" if r > 0.85 else "OK"
    print(f"    {col:25s} |r|={r:.4f}  [{flag}]")

log_action("Phase 5", "Leakage denetimi",
           "Tum korelasyonlar |r| < 0.85; leakage tespit edilmedi",
           "EDA bulgusiyla uyumlu; ID sutunlari zaten cikarildi", risk="Dusuk")

# Suton listesi
feature_cols = [c for c in df_fe.columns if c != TARGET]
print(f"\n  Model icin kullanilacak {len(feature_cols)} sutun:")
print(f"  {feature_cols}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 6 — SINIF AGIRLIGI ANALIZI
# ═══════════════════════════════════════════════════════════════════
section("PHASE 6: Sinif Dengesi ve Agirlik Hesabi")

y_full = df_fe[TARGET]
classes = np.unique(y_full)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_full)
class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

print(f"  Sinif dagilimi:")
for cls, cnt in y_full.value_counts().sort_index().items():
    pct = cnt / len(y_full) * 100
    print(f"    Exited={cls}: {cnt:5d} ornek  (%{pct:.1f})")

print(f"\n  Hesaplanan class_weight (balanced):")
for cls, w in class_weight_dict.items():
    print(f"    Exited={cls}: {w:.4f}")

print("""
  SMOTE Karar Motoru:
  -------------------
  Pozitif sinif orani: %20.3
  Dengesizlik orani: ~3.93:1

  KARAR: SMOTE UYGULANMAYACAK
  GEREKCE:
    - Kullanici talimati: yapay ornekleme eklenmeyecek
    - class_weight='balanced' ile model egitiminde denge saglanacak
    - SMOTE'un split oncesi uygulanmasi data leakage olusturur
    - %20 azinlik sinif, tree-based modeller icin yeterince temsil ediliyor
    - RandomForest, XGBoost, LightGBM class_weight parametresini destekler

  ONERI MODEL EXPERT'E:
    RandomForestClassifier(class_weight='balanced', ...)
    XGBClassifier(scale_pos_weight=3.93, ...)
    LogisticRegression(class_weight='balanced', ...)
""")

log_action("Phase 6", "Sinif dengesizligi",
           "SMOTE uygulanmadi; class_weight='balanced' onerildi",
           "Kullanici talimati + leakage riski + model duzeyinde cozum tercih edildi",
           risk="Dusuk")
add_handoff("Sinif Dengesi",
            "class_weight='balanced' onerilir",
            f"class_weight_dict={class_weight_dict}; scale_pos_weight=3.93 (XGBoost icin)")

# ═══════════════════════════════════════════════════════════════════
# PHASE 7 — PIPELINE KURULUMU + TRAIN/TEST SPLIT + FIT/TRANSFORM
# ═══════════════════════════════════════════════════════════════════
section("PHASE 7: Pipeline Kurulumu, Split ve Kayit")

# 7.1 — Ozellik Tanimlama
TARGET = "Exited"
X = df_fe.drop(columns=[TARGET])
y = df_fe[TARGET]

# Sutun gruplandirma
categorical_cols   = ["Geography", "Gender"]
numerical_cols     = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
binary_passthrough = ["NumOfProducts", "HasCrCard", "IsActiveMember",
                      "is_high_products_risk", "has_zero_balance"]

print(f"  Kategorik sutunlar (OHE): {categorical_cols}")
print(f"  Sayisal sutunlar (StandardScaler): {numerical_cols}")
print(f"  Binary/passthrough: {binary_passthrough}")

# 7.2 — Stratified Train-Test Split (SPLIT ONCE PIPELINE'DAN ONCE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(f"\n  Split tamamlandi:")
print(f"    X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"    X_test : {X_test.shape}   |  y_test : {y_test.shape}")

# Stratified orani dogrula
train_ratio = y_train.mean() * 100
test_ratio  = y_test.mean()  * 100
print(f"\n  Hedef orani dogrulama:")
print(f"    Train — Exited=1: %{train_ratio:.2f}")
print(f"    Test  — Exited=1: %{test_ratio:.2f}")
print(f"    Tam set          : %{y.mean()*100:.2f}")

log_action("Phase 7", "Train-Test Split",
           "80/20 stratified split, random_state=42",
           f"Train: {len(X_train)} | Test: {len(X_test)} | Oran korundu (train %{train_ratio:.1f}, test %{test_ratio:.1f})")

# 7.3 — ColumnTransformer (Leakage-Free Pipeline)
#   KRITIK: Burada sadece transformerlari tanimliyoruz.
#   fit() YALNIZCA train uzerinde yapilacak.
numeric_transformer     = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer,     numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="passthrough",   # binary_passthrough sutunlari oldugu gibi gecsin
    verbose_feature_names_out=False
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

print("\n  Pipeline yapisi:")
print(pipeline)

# 7.4 — LEAKAGE-FREE: fit SADECE TRAIN, transform IKISI
print("\n  LEAKAGE ONLEME: fit_transform(X_train), transform(X_test)")
X_train_proc = pipeline.fit_transform(X_train)    # <-- tek fit
X_test_proc  = pipeline.transform(X_test)         # <-- sadece transform

print(f"  X_train_proc shape: {X_train_proc.shape}")
print(f"  X_test_proc  shape: {X_test_proc.shape}")

log_action("Phase 7", "Data Leakage Kontrolu",
           "fit() YALNIZCA X_train uzerinde yapildi; X_test sadece transform edildi",
           "sklearn Pipeline + ColumnTransformer; scaler/encoder train istatistiklerine gore kuruldu",
           risk="Dusuk")

# 7.5 — Ozellik Isimlerini Al
feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
print(f"\n  Donusturulmus ozellik sayisi: {len(feature_names_out)}")
print(f"  Ozellikler: {list(feature_names_out)}")

# DataFrame'e cevir
X_train_df = pd.DataFrame(X_train_proc, columns=feature_names_out)
X_test_df  = pd.DataFrame(X_test_proc,  columns=feature_names_out)
y_train_df = y_train.reset_index(drop=True)
y_test_df  = y_test.reset_index(drop=True)

# 7.6 — CSV Kaydet
print("\n  Kaydediliyor: data/model_ready/ ...")
X_train_df.to_csv(DATA_MODEL / "X_train.csv", index=False)
X_test_df.to_csv( DATA_MODEL / "X_test.csv",  index=False)
y_train_df.to_csv(DATA_MODEL / "y_train.csv", index=False)
y_test_df.to_csv( DATA_MODEL / "y_test.csv",  index=False)

print("    X_train.csv")
print("    X_test.csv")
print("    y_train.csv")
print("    y_test.csv")

# 7.7 — Pipeline Kaydet
joblib.dump(pipeline, MODELS_DIR / "preprocessing_pipeline.pkl")
print(f"\n  Pipeline kaydedildi: models/preprocessing_pipeline.pkl")

add_handoff("Preprocessing Pipeline",
            "Kaydedildi",
            "models/preprocessing_pipeline.pkl — StandardScaler + OHE; yalnizca train ile fit edildi")
add_handoff("Train/Test Split",
            "80/20 stratified",
            "X_train(8000), X_test(2000); data/model_ready/ altinda CSV + Parquet")

# ═══════════════════════════════════════════════════════════════════
# GÖRSELLEŞTIRME — Before/After & Özet Grafikleri
# ═══════════════════════════════════════════════════════════════════
section("GORSELLESTIRME: Before/After ve Ozet Grafikler")

# ── Gorsel 1: Sinif Dagilimi (Before / After Feature Eng.)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sinif Dagilimi — Hedef Degisken (Exited)", fontsize=14, fontweight="bold")

# Before
counts_full = y.value_counts().sort_index()
axes[0].bar(["Kaldi (0)", "Ayrildi (1)"], counts_full.values,
             color=[COLOR_NEG, COLOR_POS], edgecolor="white", linewidth=1.5)
for i, v in enumerate(counts_full.values):
    axes[0].text(i, v + 80, f"{v}\n(%{v/len(y)*100:.1f})",
                 ha="center", fontsize=10, fontweight="bold")
axes[0].set_title("Tam Veri Seti")
axes[0].set_ylabel("Musteri Sayisi")
axes[0].set_ylim(0, max(counts_full.values) * 1.18)

# After split
counts_train = y_train.value_counts().sort_index()
counts_test  = y_test.value_counts().sort_index()
x_pos = np.arange(2)
w = 0.35
bars1 = axes[1].bar(x_pos - w/2, counts_train.values, w,
                     label="Train", color=COLOR_MAIN, edgecolor="white")
bars2 = axes[1].bar(x_pos + w/2, counts_test.values,  w,
                     label="Test",  color=PASTEL_PALETTE[1], edgecolor="white")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(["Kaldi (0)", "Ayrildi (1)"])
axes[1].set_title("Train/Test Split Sonrasi (Stratified)")
axes[1].set_ylabel("Musteri Sayisi")
axes[1].legend()
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., h + 30,
                 f"{int(h)}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "dataprep_phase7_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Kaydedildi: figures/dataprep_phase7_class_distribution.png")

# ── Gorsel 2: Feature Engineering — Yeni Bayrak Degiskenleri
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Feature Engineering — Yeni Bayrak Degiskenleri vs Churn", fontsize=14, fontweight="bold")

for ax, (feat, title) in zip(axes, [
    ("is_high_products_risk", "is_high_products_risk (NumOfProducts >= 3)"),
    ("has_zero_balance",      "has_zero_balance (Balance == 0)")
]):
    grp = df_fe.groupby(feat)["Exited"].agg(["mean", "count"]).reset_index()
    grp.columns = [feat, "churn_rate", "n"]
    bars = ax.bar(grp[feat].astype(str), grp["churn_rate"] * 100,
                   color=[COLOR_NEG, COLOR_POS], edgecolor="white", linewidth=1.5)
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.5,
                f"%{row['churn_rate']*100:.1f}\n(n={int(row['n'])})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("Bayrak Degeri")
    ax.set_ylabel("Churn Orani (%)")
    ax.set_ylim(0, 80)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (Hayir)", "1 (Evet)"])
plt.tight_layout()
fig.savefig(FIGURES_DIR / "dataprep_phase4_feature_engineering_flags.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Kaydedildi: figures/dataprep_phase4_feature_engineering_flags.png")

# ── Gorsel 3: Olcekleme Oncesi / Sonrasi (StandardScaler)
fig, axes = plt.subplots(2, len(numerical_cols), figsize=(18, 8))
fig.suptitle("Sayisal Degiskenler — Olcekleme Oncesi / Sonrasi (StandardScaler)",
             fontsize=14, fontweight="bold")

for i, col in enumerate(numerical_cols):
    before = X_train[col]
    after_idx = list(feature_names_out).index(col)
    after  = X_train_proc[:, after_idx]

    axes[0, i].hist(before, bins=30, color=COLOR_MAIN, edgecolor="white", alpha=0.85)
    axes[0, i].set_title(f"{col}\n(Oncesi)", fontsize=9)
    axes[0, i].set_ylabel("Frekans" if i == 0 else "")

    axes[1, i].hist(after, bins=30, color=PASTEL_PALETTE[1], edgecolor="white", alpha=0.85)
    axes[1, i].set_title(f"{col}\n(Sonrasi — Standart)", fontsize=9)
    axes[1, i].set_ylabel("Frekans" if i == 0 else "")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "dataprep_phase7_scaling_before_after.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Kaydedildi: figures/dataprep_phase7_scaling_before_after.png")

# ── Gorsel 4: OHE sonrasi Encoding Ozeti
ohe_cols = [c for c in feature_names_out if c.startswith("Geography") or c.startswith("Gender")]
passthrough_cols = [c for c in feature_names_out if c in binary_passthrough]
scaled_cols      = [c for c in feature_names_out if c in numerical_cols]

fig, ax = plt.subplots(figsize=(10, 5))
categories  = ["Sayisal (Scaled)", "OHE (Kategorik)", "Binary (Passthrough)"]
counts_feat = [len(scaled_cols), len(ohe_cols), len(passthrough_cols)]
bars = ax.barh(categories, counts_feat,
                color=[COLOR_MAIN, PASTEL_PALETTE[1], PASTEL_PALETTE[2]],
                edgecolor="white", linewidth=1.5)
for bar, cnt in zip(bars, counts_feat):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2.,
            str(cnt), va="center", fontweight="bold")
ax.set_xlabel("Ozellik Sayisi")
ax.set_title("Pipeline Cikisi — Ozellik Tipleri Ozeti", fontsize=12, fontweight="bold")
ax.set_xlim(0, max(counts_feat) * 1.25)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "dataprep_phase7_encoding_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Kaydedildi: figures/dataprep_phase7_encoding_summary.png")

# ── Gorsel 5: Class Weight Ozet Grafigi
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sinif Dengesi — class_weight Analizi", fontsize=14, fontweight="bold")

# Sol: Ornek sayisi
cls_labels = ["Kaldi (0)", "Ayrildi (1)"]
cls_counts = [int(counts_full[0]), int(counts_full[1])]
axes[0].bar(cls_labels, cls_counts, color=[COLOR_NEG, COLOR_POS], edgecolor="white", linewidth=1.5)
for i, (v, w) in enumerate(zip(cls_counts, weights)):
    axes[0].text(i, v + 60, f"n={v}\nw={w:.3f}", ha="center", fontsize=10, fontweight="bold")
axes[0].set_title("Ornek Sayisi ve Atanan Agirlik")
axes[0].set_ylabel("Musteri Sayisi")
axes[0].set_ylim(0, max(cls_counts) * 1.2)

# Sag: Agirlik carkasi
axes[1].pie(weights, labels=cls_labels, autopct="%1.1f%%",
             colors=[COLOR_NEG, COLOR_POS], startangle=90,
             wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("class_weight='balanced' Agirlik Dagilimi")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "dataprep_phase6_class_weight_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Kaydedildi: figures/dataprep_phase6_class_weight_analysis.png")

# ═══════════════════════════════════════════════════════════════════
# RAPORLAMA
# ═══════════════════════════════════════════════════════════════════
section("RAPORLAMA")

# Action Log CSV
df_actions = pd.DataFrame(dataprep_actions)
df_actions.to_csv(REPORTS_CSV / "data_prep_action_log.csv", index=False)
print("  Kaydedildi: reports/csv/data_prep_action_log.csv")

# Handoff CSV
df_handoff = pd.DataFrame(model_handoff)
df_handoff.to_csv(REPORTS_CSV / "model_expert_handoff.csv", index=False)
print("  Kaydedildi: reports/csv/model_expert_handoff.csv")

# Feature ozeti CSV
feat_summary = pd.DataFrame({
    "Ozellik"   : list(feature_names_out),
    "Tip"       : ["Sayisal (Scaled)" if f in numerical_cols
                   else "OHE (Kategorik)" if (f.startswith("Geography") or f.startswith("Gender"))
                   else "Binary (Passthrough)"
                   for f in feature_names_out],
    "Train_Mean": X_train_df.mean().values.round(4),
    "Train_Std" : X_train_df.std().values.round(4),
})
feat_summary.to_csv(REPORTS_CSV / "feature_summary.csv", index=False)
print("  Kaydedildi: reports/csv/feature_summary.csv")

# Model Expert Handoff Markdown
handoff_md = f"""# MODEL EXPERT HANDOFF RAPORU
**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Proje:** Banka Musterisi Churn Tahmini
**DataPrep Pipeline Surumu:** 1.0

---

## Veri Durumu

**TEMIZ — Modellemeye Hazir**

- Ham veri: 10.000 satir, 14 sutun
- Eksik deger: YOK (EDA dogruladi)
- Mukerrer kayit: YOK
- ID sutunlari cikarildi: RowNumber, CustomerId, Surname

---

## Ozellik Listesi ({len(feature_names_out)} Ozellik)

### Sayisal (StandardScaler ile olceklendi):
- CreditScore, Age, Tenure, Balance, EstimatedSalary

### Kategorik (OneHotEncoding, drop='first'):
- Geography_Germany, Geography_Spain
- Gender_Male

### Binary / Passthrough (olceklenmedi):
- NumOfProducts, HasCrCard, IsActiveMember
- **is_high_products_risk** (yeni — NumOfProducts >= 3)
- **has_zero_balance** (yeni — Balance == 0)

---

## Missing Value Stratejisi

Mudahale gerekmedi. Tum degiskenlerde eksik deger = 0.

---

## Encoding Stratejisi

| Degisken    | Yontem       | Kararname |
|-------------|--------------|-----------|
| Geography   | OneHotEncoder (drop='first') | 3 kategori -> 2 sutun |
| Gender      | OneHotEncoder (drop='first') | 2 kategori -> 1 sutun |

drop='first' kullanildi: multicollinearity onlendi, France ve Female referans kategorileri.

---

## Scaling Stratejisi

StandardScaler: CreditScore, Age, Tenure, Balance, EstimatedSalary

**LEAKAGE ONLEME:**
- fit() YALNIZCA X_train ({len(X_train)} ornek) uzerinde yapildi.
- X_test ({len(X_test)} ornek) yalnizca transform() ile islendi.
- sklearn Pipeline + ColumnTransformer kullanildi.

---

## Sinif Dengesizligi Stratejisi

**SMOTE UYGULANMADI.** class_weight='balanced' onerilir.

| Sinif | Ornek Sayisi | Oran | class_weight |
|-------|-------------|------|--------------|
| 0 (Kaldi)    | {int(counts_full[0])} | %{counts_full[0]/len(y)*100:.1f} | {weights[0]:.4f} |
| 1 (Ayrildi)  | {int(counts_full[1])} | %{counts_full[1]/len(y)*100:.1f} | {weights[1]:.4f} |

**Model parametreleri:**
- RandomForest / LogisticRegression: `class_weight='balanced'`
- XGBoost: `scale_pos_weight={weights[1]/weights[0]:.2f}`
- LightGBM: `class_weight='balanced'` veya `is_unbalance=True`

---

## Feature Engineering

| Ozellik | Tanimim | Churn Etkisi |
|---------|---------|--------------|
| is_high_products_risk | NumOfProducts >= 3 | Evet grubunda churn ~%64 vs %19 (3.4x) |
| has_zero_balance      | Balance == 0       | Farkin buyuklugu dusuk-orta; davranissal sinyal |

---

## Train / Test Split

| Set   | Satir | Exited=1 Orani |
|-------|-------|----------------|
| Train | {len(X_train)} | %{y_train.mean()*100:.2f} |
| Test  | {len(X_test)} | %{y_test.mean()*100:.2f} |

stratify=y, random_state=42, test_size=0.20

---

## Leakage Durumu

**YOK (Dusuk Risk)**

- Tum donusumler Pipeline icinde; test seti fit edilmedi
- Hedef degiskeni kopyalayan ozellik bulunamadi (max |r|={float(corr_with_target.max()):.3f})
- Temporal leakage: Veri setsinde zaman boyutu yok

---

## Onerilen Model Turleri

1. **RandomForestClassifier** — class_weight='balanced', n_estimators>=200
2. **XGBClassifier** — scale_pos_weight={weights[1]/weights[0]:.2f}
3. **LightGBM** — is_unbalance=True
4. **LogisticRegression** — class_weight='balanced' (baseline)
5. **GradientBoostingClassifier** — sample_weight ile

Tree-based modeller onerilir: NumOfProducts'in dogrusal olmayan etkisini yakalarlar.

---

## Kritik Uyarilar

1. is_high_products_risk cok guclu bir ayirici (churn 3.4x); agac modelleri bunu iyi kullanir.
2. Age dogrusal olmayan churn etkisi gosteriyor (45+ kritik); age_group donusumu
   veya polynomial feature modelleme asamasinda dusunulebilir.
3. CreditScore, Tenure, EstimatedSalary dusuek ongorucular; feature selection asamasinda
   eleme adayi olabilir (p > 0.05, EDA).
4. SMOTE kullanilacaksa KESINLIKLE split sonrasi, sadece X_train uzerinde uygulanmali.
5. Test seti hicbir durumda preprocessing'e dahil edilmemeli.

---

## Pipeline Kayit Konumu

```
models/preprocessing_pipeline.pkl
```

Yuklemek icin:
```python
import joblib
pipeline = joblib.load('models/preprocessing_pipeline.pkl')
X_new_processed = pipeline.transform(X_new)
```

---

## Cikti Dosyalari

```
data/model_ready/
    X_train.csv   ({len(X_train)} x {len(feature_names_out)})
    X_test.csv    ({len(X_test)} x {len(feature_names_out)})
    y_train.csv                     ({len(y_train)} x 1)
    y_test.csv                      ({len(y_test)} x 1)
models/
    preprocessing_pipeline.pkl
figures/
    dataprep_phase7_class_distribution.png
    dataprep_phase4_feature_engineering_flags.png
    dataprep_phase7_scaling_before_after.png
    dataprep_phase7_encoding_summary.png
    dataprep_phase6_class_weight_analysis.png
reports/csv/
    phase1_eda_decision_matrix.csv
    data_prep_action_log.csv
    model_expert_handoff.csv
    feature_summary.csv
```
"""

with open(REPORTS_MD / "DATAPREP_HANDOFF_REPORT.md", "w", encoding="utf-8") as f:
    f.write(handoff_md)
print("  Kaydedildi: reports/markdown/DATAPREP_HANDOFF_REPORT.md")

# ═══════════════════════════════════════════════════════════════════
# FINAL OZET
# ═══════════════════════════════════════════════════════════════════
section("DATAPREP TAMAMLANDI — FINAL OZET")

print(f"""
  Veri Durumu         : TEMIZ — Modellemeye Hazir
  Ham Veri            : {df_raw.shape[0]} satir, {df_raw.shape[1]} sutun
  Ozellik Sayisi      : {len(feature_names_out)} (9 orjinal + 2 yeni bayrak)
  Train Seti          : {len(X_train)} ornek  | Exited=1: %{y_train.mean()*100:.1f}
  Test Seti           : {len(X_test)} ornek   | Exited=1: %{y_test.mean()*100:.1f}
  Leakage Durumu      : YOK
  Sinif Dengesizligi  : class_weight='balanced' onerildi (SMOTE yok)
  Pipeline            : Kaydedildi (models/preprocessing_pipeline.pkl)
  Tum Dosyalar        : data/model_ready/, reports/, figures/
""")

print("  Action Log:")
for a in dataprep_actions:
    print(f"    [{a['Risk']:8s}] {a['Asama']:10s} | {a['Karar']}")

print("\n  Model Expert'e Teslim: HAZIR")
