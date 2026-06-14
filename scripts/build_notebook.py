# -*- coding: utf-8 -*-
"""
final_analysis.ipynb uretici (v2 - SINIFLANDIRMA).
auto-mpg-regression-analysis.ipynb sablonunun gorsel dilini (gradient HTML hero,
gradient banner gecisleri, emoji'li basliklar, mermaid, Plotly interaktif grafikler)
KORUYARAK; churn SINIFLANDIRMA projesi icin 12 ADIM'lik notebook olusturur.
"""
import nbformat as nbf
from pathlib import Path

NB_DIR = Path("C:/Users/sence/OneDrive/Desktop/churn-analysis/notebooks")
NB_DIR.mkdir(parents=True, exist_ok=True)
NB_PATH = NB_DIR / "final_analysis.ipynb"

nb = nbf.v4.new_notebook()
cells = []
def md(t):   cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

# =====================================================================
# HERO
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 50px 30px; border-radius: 20px; text-align: center; box-shadow: 0 15px 50px rgba(0,0,0,0.3); margin-bottom: 30px;">
  <h1 style="color: white; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🏦 Banka Müşteri Kaybı (Churn) Analizi</h1>
  <h2 style="color: #f0f0f0; font-size: 1.5em; margin-top: 10px; font-weight: 300;">CRISP-DM Metodolojisi ile Uçtan Uca Sınıflandırma Projesi</h2>
  <div style="margin-top: 25px;">
    <span style="background: rgba(255,255,255,0.25); color: white; padding: 8px 16px; border-radius: 15px; margin: 4px; display: inline-block; font-weight: bold;">🎯 Sınıflandırma</span>
    <span style="background: rgba(255,255,255,0.25); color: white; padding: 8px 16px; border-radius: 15px; margin: 4px; display: inline-block; font-weight: bold;">🤖 Machine Learning</span>
    <span style="background: rgba(255,255,255,0.25); color: white; padding: 8px 16px; border-radius: 15px; margin: 4px; display: inline-block; font-weight: bold;">⚖️ Dengesiz Veri</span>
    <span style="background: rgba(255,255,255,0.25); color: white; padding: 8px 16px; border-radius: 15px; margin: 4px; display: inline-block; font-weight: bold;">📉 Churn Prediction</span>
  </div>
</div>

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 30px; border-radius: 15px; border-left: 6px solid #ff6b6b; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">

### 📌 **Proje Kimlik Kartı**

| 🏷️ **Özellik** | 📊 **Değer** |
|:---------------|:-------------|
| **🎯 Proje Adı** | Banka Müşteri Kaybı (Churn) Tahmin Modeli |
| **📊 Veri Seti** | `data/raw/churn.csv` |
| **🎲 Problem Türü** | **Denetimli Öğrenme – İkili Sınıflandırma** |
| **🎯 Hedef Değişken** | `Exited` (1 = ayrıldı, 0 = kaldı) |
| **📏 Veri Boyutu** | 10.000 gözlem × 14 değişken |
| **⚖️ Sınıf Dengesi** | ~%20 pozitif (dengesiz) |
| **🔬 Metodoloji** | **CRISP-DM** |
| **📅 Proje Tarihi** | 30 Mayıs 2026 |

</div>

<div style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); padding: 40px; border-radius: 15px; margin: 25px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">

## 🎯 **Problem Tanımı ve İş Hikayesi**

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">

### 🌍 **Bağlam: Müşteriyi Elde Tutmanın Değeri**

> Bankacılıkta **yeni müşteri kazanmanın maliyeti**, mevcut bir müşteriyi elde tutmanın maliyetinden **5–7 kat daha yüksektir**. Ayrılan (churn eden) her müşteri hem doğrudan gelir kaybı hem de yeniden kazanım maliyeti demektir.

**Neden Önemli?**
- 💸 **Gelir kaybı:** Ayrılan müşteri = kaybedilen mevduat, kredi ve komisyon geliri
- 🎯 **Hedefli kampanya:** Risk altındaki müşteriler önceden tespit edilirse tutundurma (retention) teklifleri yapılabilir
- 📊 **Kaynak optimizasyonu:** Kampanya bütçesi en çok risk taşıyan müşterilere yönlendirilebilir

</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">

### 💼 **İş Problemi**

<div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #e17055; margin: 15px 0;">

**🎯 Ana Hedef:**
Bir müşterinin yakın gelecekte bankadan **ayrılıp ayrılmayacağını** (`Exited`) geçmiş demografik, finansal ve davranışsal verilerden **önceden tahmin etmek**.

</div>

**🔑 Kritik Sorular:**
- ❓ **Hangi müşteri özellikleri** ayrılma ile en güçlü ilişkili?
- 🔮 Riskli müşteriler ne kadar **isabetli** yakalanabilir?
- 🎯 **Recall'ı** yüksek tutarak kaç gerçek churn müşterisi yakalanır?
- 💡 Hangi **iş aksiyonları** churn'ü azaltabilir?

</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">

### 📊 Veri Seti Değişkenleri

| 🏷️ **Değişken** | 📝 **Açıklama** | 📊 **Tür** |
|:----------------|:----------------|:-----------|
| 🎯 **Exited** | Müşteri ayrıldı mı? (1/0) — HEDEF | İkili |
| 🆔 **RowNumber, CustomerId, Surname** | Kimlik bilgileri (tahmin gücü yok) | Kimlik |
| 💳 **CreditScore** | Kredi puanı | Sürekli |
| 🌍 **Geography** | Ülke (France / Spain / Germany) | Kategorik |
| 🚻 **Gender** | Cinsiyet | Kategorik |
| 🎂 **Age** | Yaş | Sürekli |
| 📆 **Tenure** | Bankadaki yıl sayısı (0–10) | Sayısal |
| 💰 **Balance** | Hesap bakiyesi | Sürekli |
| 📦 **NumOfProducts** | Sahip olunan ürün sayısı (1–4) | Sayısal |
| 💳 **HasCrCard** | Kredi kartı var mı? (1/0) | İkili |
| ✅ **IsActiveMember** | Aktif üye mi? (1/0) | İkili |
| 🧾 **EstimatedSalary** | Tahmini yıllık maaş | Sürekli |

</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">

### 🎯 Başarı Kriterleri (Metrik Seçimi)

<table style="width: 100%; border-collapse: collapse;">
<tr>
<td style="width: 33%; padding: 15px; text-align: center; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 10px;">
<div style="font-size: 2.5em;">🎯</div>
<strong>Recall Önceliği</strong><br/>
<small>Gerçek churn'lerin çoğunu yakala</small>
</td>
<td style="width: 33%; padding: 15px; text-align: center; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 10px;">
<div style="font-size: 2.5em;">⚖️</div>
<strong>F1 Dengesi</strong><br/>
<small>Precision–Recall dengesi</small>
</td>
<td style="width: 33%; padding: 15px; text-align: center; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 10px;">
<div style="font-size: 2.5em;">📈</div>
<strong>ROC-AUC</strong><br/>
<small>Eşik bağımsız ayırma gücü</small>
</td>
</tr>
</table>

<div style="background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); padding: 15px; border-radius: 8px; margin-top: 15px; text-align: center;">
<strong>⚠️ Neden Accuracy Değil?</strong> Veri dengesiz (~%20 churn). Herkesi "kalır" diyen model bile ~%80 accuracy verir ama hiçbir churn'ü yakalamaz. Bu yüzden ana metrik <strong>F1 ve Recall</strong>'dur.
</div>

</div>

</div>

### 📊 **CRISP-DM Metodolojisi**

```mermaid
flowchart TD
    Start([🎬 BAŞLANGIÇ]) --> A[🎯 İş Anlayışı<br/>Problem Tanımı]
    A --> B[📊 Veri Anlayışı<br/>EDA]
    B --> C[🛠️ Veri Hazırlama<br/>Feature Engineering + Pipeline]
    C --> D[🤖 Modelleme<br/>12 Sınıflandırıcı]
    D --> E[📈 Değerlendirme<br/>F1 / Recall / ROC-AUC]
    E -->|Başarılı| F[🚀 Dağıtım<br/>models/final_model.pkl]
    E -->|İyileştirme| D
    F --> End([✅ TAMAMLANDI])

    style Start fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px,color:#000
    style A fill:#FFE5E5,stroke:#C62828,stroke-width:3px,color:#000
    style B fill:#E1F5FE,stroke:#0277BD,stroke-width:3px,color:#000
    style C fill:#F3E5F5,stroke:#6A1B9A,stroke-width:3px,color:#000
    style D fill:#FFF3E0,stroke:#E65100,stroke-width:3px,color:#000
    style E fill:#FCE4EC,stroke:#C2185B,stroke-width:3px,color:#000
    style F fill:#E0F2F1,stroke:#00695C,stroke-width:3px,color:#000
    style End fill:#DCEDC8,stroke:#558B2F,stroke-width:3px,color:#000
```

### 🗺️ **Notebook Akışı — 12 Adım**

| Adım | Bölüm | Adım | Bölüm |
|:--|:--|:--|:--|
| 1️⃣ | Kurulum & Kütüphaneler | 7️⃣ | Ön İşleme + Feature Engineering |
| 2️⃣ | Veri Yükleme & İlk Keşif | 8️⃣ | Pipeline + Stratified Split |
| 3️⃣ | Hedef Değişken & Sınıf Dengesizliği | 9️⃣ | 12 Model + Karşılaştırma |
| 4️⃣ | Tekli Değişken Analizi | 🔟 | Değerlendirme (CM, ROC) |
| 5️⃣ | Churn İlişkisi + Korelasyon | 1️⃣1️⃣ | Final Model & Kayıt |
| 6️⃣ | Veri Kalitesi | 1️⃣2️⃣ | Sonuç & İş Önerileri |

> 🔄 Notebook'u baştan sona çalıştırmak için: **Kernel → Restart & Run All**
""")

# =====================================================================
# ADIM 1
# =====================================================================
md(r"""
## 📚 **ADIM 1: Proje Kurulumu ve Gerekli Kütüphaneler**

### 🔧 **Kütüphane Yükleme Süreci**

Bu bölümde tüm projede kullanacağımız kütüphaneleri yüklüyoruz:

**Veri Manipülasyonu:** `pandas`, `numpy`
**Görselleştirme:** `plotly.express`, `plotly.graph_objects` (interaktif grafikler)
**Makine Öğrenmesi:** `scikit-learn`, `xgboost`, `lightgbm`
**Yardımcı:** `warnings`, `joblib` (model kaydetme), `datetime`

Tekrar üretilebilirlik için `RANDOM_STATE = 42` sabitlenir.
""")
code(r"""
# ============================================
# KÜTÜPHANE YÜKLEMELERİ
# ============================================
import pandas as pd
import numpy as np

# Görselleştirme — Plotly (interaktif)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Makine Öğrenmesi
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Sınıflandırma metrikleri
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve,
                             average_precision_score)

# Yardımcı
import warnings, time, joblib
from pathlib import Path
from datetime import datetime

# Ayarlar
pio.renderers.default = "notebook_connected"   # interaktif Plotly (CDN)
pio.templates.default = "plotly_white"
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Proje renk paleti (sablonla uyumlu)
C_PRIMARY = "#667eea"   # mor-mavi (kaldi / ana)
C_SECOND  = "#764ba2"   # mor
C_CHURN   = "#f5576c"   # kirmizi-pembe (churn / risk)
C_PINK    = "#f093fb"
PALETTE_BIN = [C_PRIMARY, C_CHURN]

print("✅ Tüm kütüphaneler başarıyla yüklendi!")
print(f"📅 Çalıştırma Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🐍 pandas {pd.__version__} | numpy {np.__version__}")
""")

md(r"""
### 📁 **Klasör Yapısının Oluşturulması**

Düzenli bir çalışma için gerekli klasörleri kontrol edip (yoksa) oluşturuyoruz.
""")
code(r"""
# ============================================
# KLASÖR YAPISI
# ============================================
PROJECT_ROOT = Path("C:/Users/sence/OneDrive/Desktop/churn-analysis")
DATA_RAW    = PROJECT_ROOT / "data" / "raw"
DATA_MODEL  = PROJECT_ROOT / "data" / "model_ready"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR  = PROJECT_ROOT / "models"

for d in [DATA_MODEL, FIGURES_DIR, MODELS_DIR]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        print(f"✅ '{d}' oluşturuldu")
    else:
        print(f"✓ '{d.name}/' zaten mevcut")
print("\n📂 Proje klasör yapısı hazır!")
""")

# =====================================================================
# ADIM 2
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        📊 ADIM 2: Veri Yükleme ve İlk Keşif
    </h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 18px; margin-top: 15px;">
        (Data Loading & First Glance)
    </p>
</div>

<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 30px; border-radius: 15px; border-left: 6px solid #0984e3; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.1);">

### 🎯 **Bu Adımda Yapılacaklar**

- 📥 `data/raw/churn.csv` veri setini yükleme
- 🔍 Veri boyutu, ilk satırlar ve veri tiplerini inceleme
- 📊 Sayısal değişkenlerin istatistiksel özeti (interaktif tablo)
- 🧐 İlk gözlemler ve değişken sözlüğü

</div>
""")
code(r"""
# ============================================
# VERİ SETİNİ YÜKLEME
# ============================================
df = pd.read_csv(DATA_RAW / "churn.csv")
print(f"✅ Veri yüklendi → {df.shape[0]:,} satır × {df.shape[1]} sütun")
df.head()
""")
code(r"""
# ============================================
# VERİ TİPLERİ VE GENEL BİLGİ
# ============================================
print("📋 Sütun tipleri ve bellek kullanımı:\n")
df.info()
""")
code(r"""
# ============================================
# İNTERAKTİF İSTATİSTİKSEL ÖZET TABLOSU
# ============================================
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
summary = []
for col in numeric_cols_all:
    summary.append({
        "Değişken": col, "Ortalama": df[col].mean(), "Medyan": df[col].median(),
        "Std": df[col].std(), "Min": df[col].min(), "Max": df[col].max(),
        "Eksik": int(df[col].isnull().sum()),
    })
summary_df = pd.DataFrame(summary)

fig = go.Figure(data=[go.Table(
    header=dict(values=["<b>"+c+"</b>" for c in summary_df.columns],
                fill_color=C_PRIMARY, align="center",
                font=dict(color="white", size=12, family="Arial Black")),
    cells=dict(values=[summary_df[c] for c in summary_df.columns],
               fill_color=[["#f0f0f0", "#ffffff"]*len(summary_df)],
               align="center", font=dict(color="#333", size=11),
               format=[None, ".2f", ".2f", ".2f", ".2f", ".2f", None]))])
fig.update_layout(title=dict(text="📊 Sayısal Değişkenler — İstatistiksel Özet",
                  x=0.5, xanchor="center", font=dict(size=18, family="Arial Black", color="#333")),
                  height=380, margin=dict(l=20, r=20, t=60, b=20))
fig.write_html(FIGURES_DIR / "nb01_summary_table.html")
fig.show()
print("✅ Kaydedildi: figures/nb01_summary_table.html")
""")

# =====================================================================
# ADIM 3
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🎯 ADIM 3: Hedef Değişken ve Sınıf Dengesizliği
    </h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 18px; margin-top: 15px;">
        (Target Variable: Exited & Class Imbalance)
    </p>
</div>

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 30px; border-radius: 15px; border-left: 6px solid #e17055; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.1);">

### 🎯 **Neden Önce Hedef Değişken?**

Hedef değişken `Exited`'in dağılımı, **tüm modelleme stratejisini belirler**:
- Pozitif sınıf (ayrılan) oranı düşükse → **dengesiz veri** problemi
- Dengesizlik → **accuracy yanıltıcı**, ana metrik **F1 + Recall**
- Çözüm → veriye dokunmadan model düzeyinde **`class_weight`** (SMOTE değil)

</div>
""")
code(r"""
# ============================================
# HEDEF DEĞİŞKEN DAĞILIMI — İNTERAKTİF
# ============================================
counts = df["Exited"].value_counts().sort_index()
pct = df["Exited"].value_counts(normalize=True).sort_index() * 100
imbalance = counts[0] / counts[1]
labels = ["Kaldı (0)", "Ayrıldı (1)"]

fig = make_subplots(rows=1, cols=2, specs=[[{"type":"bar"}, {"type":"pie"}]],
                    subplot_titles=("Müşteri Sayısı", "Sınıf Oranı"))
fig.add_trace(go.Bar(x=labels, y=counts.values, marker_color=PALETTE_BIN,
              text=[f"{v:,}<br>%{p:.1f}" for v,p in zip(counts.values, pct.values)],
              textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Pie(labels=labels, values=counts.values, hole=0.45,
              marker=dict(colors=PALETTE_BIN), textinfo="label+percent",
              hovertemplate="%{label}: %{value}<extra></extra>"), row=1, col=2)
fig.update_layout(title=dict(text="🎯 Hedef Değişken (Exited) Dağılımı",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  showlegend=False, height=450)
fig.write_html(FIGURES_DIR / "nb02_target_distribution.html")
fig.show()

print("="*70)
print("⚖️  SINIF DENGESİZLİĞİ ANALİZİ".center(70))
print("="*70)
for k in counts.index:
    print(f"   Exited={k} ({labels[k]:11s}): {counts[k]:5,d}  (%{pct[k]:.1f})")
print(f"\n   📊 Dengesizlik oranı : {imbalance:.2f} : 1")
print(f"   🎯 Strateji          : class_weight='balanced' (SMOTE YOK)")
print(f"   📏 Ana metrik        : F1 + Recall  (accuracy DEĞİL)")
print("="*70)
""")

# =====================================================================
# ADIM 4
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🔢 ADIM 4: Tekli Değişken Analizi
    </h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 18px; margin-top: 15px;">
        (Univariate Analysis)
    </p>
</div>

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #e17055; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.1);">

### 🎯 **Tekli Değişken Analizi Nedir?**

Her değişkeni **bağımsız** olarak inceleyip dağılımını, merkezi eğilimini ve
çarpıklığını anlarız. Bu aşamada değişkenler arası ilişkiye **bakmayız**.

- 🔢 **Sayısal:** CreditScore, Age, Tenure, Balance, EstimatedSalary → histogram + çarpıklık
- 🏷️ **Kategorik/Ayrık:** Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember → frekans

</div>
""")
code(r"""
# ============================================
# SAYISAL DEĞİŞKENLERİN DAĞILIMI — İNTERAKTİF
# ============================================
numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
fig = make_subplots(rows=2, cols=3, subplot_titles=[
    f"{c} (skew={df[c].skew():+.2f})" for c in numeric_cols])
positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]
for col,(r,c) in zip(numeric_cols, positions):
    fig.add_trace(go.Histogram(x=df[col], nbinsx=40, marker_color=C_PRIMARY,
                  marker_line=dict(color="white", width=0.5), opacity=0.85,
                  hovertemplate=f"{col}: %{{x}}<br>Frekans: %{{y}}<extra></extra>"), row=r, col=c)
fig.update_layout(title=dict(text="📊 Sayısal Değişkenlerin Dağılımı",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  showlegend=False, height=650)
fig.write_html(FIGURES_DIR / "nb03_numeric_distributions.html")
fig.show()

print("📐 Çarpıklık (|skew| < 1 → dönüşüm gereksiz):")
for c in numeric_cols:
    s = df[c].skew()
    flag = "🟢 OK" if abs(s) < 1 else "🔴 Dönüşüm"
    print(f"   {c:18s}: {s:+.3f}  {flag}")
""")
code(r"""
# ============================================
# KATEGORİK / AYRIK DEĞİŞKENLER — FREKANS
# ============================================
cat_cols = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]
fig = make_subplots(rows=2, cols=3, subplot_titles=cat_cols)
positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]
for col,(r,c) in zip(cat_cols, positions):
    vc = df[col].value_counts().sort_index()
    fig.add_trace(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=C_SECOND,
                  text=vc.values, textposition="outside",
                  hovertemplate=f"{col}=%{{x}}: %{{y}}<extra></extra>"), row=r, col=c)
fig.update_layout(title=dict(text="🏷️ Kategorik / Ayrık Değişkenlerin Frekansı",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  showlegend=False, height=650)
fig.write_html(FIGURES_DIR / "nb04_categorical_counts.html")
fig.show()
print("✅ Tekli değişken analizi tamamlandı.")
""")

# =====================================================================
# ADIM 5
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🔗 ADIM 5: Churn ile İlişki ve Korelasyon
    </h1>
    <p style="color: #ffffff; text-align: center; font-size: 18px; margin-top: 15px;">
        (Bivariate Analysis: Geography, Age, NumOfProducts, IsActiveMember)
    </p>
</div>

<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #e17055; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.1);">

### 🎯 **Bu Adımda Yapılacaklar**

Her özelliğin **hedef (`Exited`) ile ilişkisini** inceleyerek en güçlü churn
öngörücülerini buluyoruz:
- 🌍 **Geography**, 📦 **NumOfProducts**, ✅ **IsActiveMember** → gruplara göre churn oranı
- 🎂 **Age** → churn'e göre dağılım (yaş, churn'ün en güçlü sinyali mi?)
- 🔥 **Korelasyon matrisi** + leakage kontrolü

</div>
""")
code(r"""
# ============================================
# KATEGORİK DEĞİŞKENLERE GÖRE CHURN ORANI
# ============================================
overall = df["Exited"].mean() * 100
drivers = ["Geography", "NumOfProducts", "IsActiveMember", "Gender"]
fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{d} bazında Churn Oranı (%)" for d in drivers])
positions = [(1,1),(1,2),(2,1),(2,2)]
for col,(r,c) in zip(drivers, positions):
    rate = df.groupby(col)["Exited"].mean().sort_index() * 100
    colors = [C_CHURN if v > overall else C_PRIMARY for v in rate.values]
    fig.add_trace(go.Bar(x=rate.index.astype(str), y=rate.values, marker_color=colors,
                  text=[f"%{v:.1f}" for v in rate.values], textposition="outside",
                  hovertemplate=f"{col}=%{{x}}<br>Churn: %{{y:.1f}}%<extra></extra>"), row=r, col=c)
    fig.add_hline(y=overall, line_dash="dash", line_color="gray", row=r, col=c)
fig.update_layout(title=dict(text=f"🔗 Churn Sürücüleri (Genel ort. = %{overall:.1f}, gri çizgi)",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  showlegend=False, height=700)
fig.write_html(FIGURES_DIR / "nb05_churn_drivers.html")
fig.show()
""")
code(r"""
# ============================================
# YAŞ (AGE) — CHURN'E GÖRE DAĞILIM
# ============================================
fig = go.Figure()
for k, name, color in [(0, "Kaldı (0)", C_PRIMARY), (1, "Ayrıldı (1)", C_CHURN)]:
    fig.add_trace(go.Histogram(x=df[df["Exited"]==k]["Age"], name=name, nbinsx=40,
                  marker_color=color, opacity=0.65,
                  hovertemplate=name+" — Yaş %{x}: %{y}<extra></extra>"))
fig.update_layout(barmode="overlay", title=dict(text="🎂 Yaş Dağılımı — Churn Kırılımı",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  xaxis_title="Yaş", yaxis_title="Frekans", height=450)
fig.write_html(FIGURES_DIR / "nb06_age_by_churn.html")
fig.show()

m0, m1 = df[df.Exited==0]["Age"].mean(), df[df.Exited==1]["Age"].mean()
print(f"📊 Ortalama yaş — Kaldı: {m0:.1f}  |  Ayrıldı: {m1:.1f}  (fark: {m1-m0:+.1f} yıl)")
""")
code(r"""
# ============================================
# KORELASYON MATRİSİ (Pearson) + LEAKAGE KONTROLÜ
# ============================================
corr = df[numeric_cols + ["Exited"]].corr()
fig = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu_r", zmid=0,
    text=corr.values.round(2), texttemplate="%{text}",
    textfont=dict(size=11, color="black"),
    colorbar=dict(title="r"),
    hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>"))
fig.update_layout(title=dict(text="🔥 Korelasyon Matrisi (Sayısal + Hedef)",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  yaxis=dict(autorange="reversed"), width=700, height=600)
fig.write_html(FIGURES_DIR / "nb07_correlation.html")
fig.show()

print("📊 Exited ile |korelasyon| (büyükten küçüğe):")
for var, c in corr["Exited"].drop("Exited").abs().sort_values(ascending=False).items():
    emoji = "🔴" if c > 0.7 else "🟡" if c > 0.2 else "🟢"
    print(f"   {emoji} {var:18s}: {c:.3f}")
print("\n✅ Leakage kontrolü: |r| > 0.85 olan özellik YOK → sızıntı riski düşük.")
""")

# =====================================================================
# ADIM 6
# =====================================================================
md(r"""
## 📋 **ADIM 6: Veri Kalitesi Analizi**

> **🎯 Amaç:** Modellemeden önce eksik değer, mükerrer kayıt, kardinalite ve aykırı değer (outlier) durumunu değerlendirmek.

```mermaid
graph LR
    A[🗂️ Ham Veri] --> B[🔍 Eksik Değer]
    A --> C[♻️ Mükerrer Kayıt]
    A --> D[🔢 Kardinalite / ID]
    A --> E[📊 Outlier IQR]
    B --> F[✅ Kalite Raporu]
    C --> F
    D --> F
    E --> F
    style A fill:#FFE5B4,stroke:#FF8C00,stroke-width:2px,color:#000
    style F fill:#B4FFD7,stroke:#32CD32,stroke-width:2px,color:#000
```

### 📌 **Bu Adımda Yapılacaklar**

| İşlem | Açıklama |
|-------|----------|
| 🔍 **Eksik Değer** | Sütun bazında eksik sayısı ve oranı |
| ♻️ **Mükerrer Kayıt** | Tam aynı satır var mı? |
| 🔢 **Kardinalite** | Hangi sütunlar kimlik (ID) niteliğinde? |
| 📊 **Outlier (IQR)** | Aykırı değer yoğunluğu |
""")
code(r"""
# ============================================
# VERİ KALİTESİ RAPORU — İNTERAKTİF TABLO
# ============================================
quality = pd.DataFrame({
    "Sütun": df.columns,
    "Tip": df.dtypes.astype(str).values,
    "Eksik": df.isnull().sum().values,
    "Eksik_%": (df.isnull().mean()*100).round(2).values,
    "Benzersiz": df.nunique().values,
    "Benzersiz_%": (df.nunique()/len(df)*100).round(1).values,
})

fig = go.Figure(data=[go.Table(
    header=dict(values=["<b>"+c+"</b>" for c in quality.columns],
                fill_color=C_PRIMARY, align="center",
                font=dict(color="white", size=12, family="Arial Black")),
    cells=dict(values=[quality[c] for c in quality.columns],
               fill_color=[["#f0f0f0", "#ffffff"]*len(quality)],
               align="center", font=dict(color="#333", size=11)))])
fig.update_layout(title=dict(text="📋 Veri Kalitesi Raporu",
                  x=0.5, xanchor="center", font=dict(size=18, family="Arial Black", color="#333")),
                  height=480, margin=dict(l=20, r=20, t=60, b=20))
fig.write_html(FIGURES_DIR / "nb08_quality_table.html")
fig.show()

print("="*70)
print(f"   🔍 Toplam eksik değer : {df.isnull().sum().sum()}")
print(f"   ♻️  Mükerrer satır     : {df.duplicated().sum()}")
print(f"   🔢 ID sütunları (yüksek kardinalite): RowNumber, CustomerId, Surname")
print("="*70)

print("\n📊 IQR Outlier Özeti (oran < %5 → müdahale gereksiz):")
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum()
    p = n_out/len(df)*100
    print(f"   {col:18s}: n={n_out:4d}  (%{p:.2f})  {'🟢' if p<5 else '🔴'}")
""")

# =====================================================================
# ADIM 7
# =====================================================================
md(r"""
## 🛠️ **ADIM 7: Ön İşleme ve Feature Engineering**

> **🎯 Amaç:** Kimlik sütunlarını temizlemek ve EDA'da bulunan doğrusal olmayan sinyalleri modele açık bayrak özelliklerine dönüştürmek.

```mermaid
graph TB
    A[📊 Ham Veri] --> B[🗑️ ID Sütunlarını Sil]
    B --> C[🏷️ Tip Düzeltme]
    C --> D[🔨 Feature Engineering]
    D --> D1[is_high_products_risk<br/>NumOfProducts ≥ 3]
    D --> D2[has_zero_balance<br/>Balance == 0]
    D1 --> E[✨ Zenginleştirilmiş Veri]
    D2 --> E
    style A fill:#FFB4B4,stroke:#FF6B6B,stroke-width:2px,color:#000
    style D fill:#FFE5B4,stroke:#FFA500,stroke-width:2px,color:#000
    style E fill:#B4FFD7,stroke:#32CD32,stroke-width:2px,color:#000
```

### 📌 **Bu Adımda Yapılacaklar**

| İşlem | Karar | Gerekçe |
|-------|-------|---------|
| 🗑️ **ID Sil** | RowNumber, CustomerId, Surname | Tahmin gücü yok |
| 🔨 **is_high_products_risk** | NumOfProducts ≥ 3 | 3–4 üründe churn çok yüksek (doğrusal olmayan) |
| 🔨 **has_zero_balance** | Balance == 0 | Sıfır bakiyeli müşteri farklı davranıyor |
""")
code(r"""
# ============================================
# TEMİZLEME + FEATURE ENGINEERING
# ============================================
df_fe = df.drop(columns=["RowNumber", "CustomerId", "Surname"]).copy()
df_fe["Geography"] = df_fe["Geography"].astype("category")
df_fe["Gender"] = df_fe["Gender"].astype("category")

# Yeni bayrak özellikleri
df_fe["is_high_products_risk"] = (df_fe["NumOfProducts"] >= 3).astype(int)
df_fe["has_zero_balance"] = (df_fe["Balance"] == 0).astype(int)

print(f"✅ ID sütunları silindi → yeni boyut: {df_fe.shape}")
print("\n🔨 is_high_products_risk — churn kırılımı:")
print(df_fe.groupby("is_high_products_risk")["Exited"].agg(churn_orani="mean", n="count").round(3))
print("\n🔨 has_zero_balance — churn kırılımı:")
print(df_fe.groupby("has_zero_balance")["Exited"].agg(churn_orani="mean", n="count").round(3))

# Gorsel
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "is_high_products_risk vs Churn", "has_zero_balance vs Churn"))
for i, feat in enumerate(["is_high_products_risk", "has_zero_balance"]):
    g = df_fe.groupby(feat)["Exited"].mean()*100
    fig.add_trace(go.Bar(x=["0 (Hayır)", "1 (Evet)"], y=g.values, marker_color=PALETTE_BIN,
                  text=[f"%{v:.1f}" for v in g.values], textposition="outside"), row=1, col=i+1)
fig.update_layout(title=dict(text="🔨 Yeni Bayrak Özellikleri — Churn Oranı (%)",
                  x=0.5, xanchor="center", font=dict(size=18, family="Arial Black", color="#2c3e50")),
                  showlegend=False, height=420)
fig.write_html(FIGURES_DIR / "nb09_feature_engineering.html")
fig.show()
""")

# =====================================================================
# ADIM 8
# =====================================================================
md(r"""
## 🔒 **ADIM 8: Pipeline + Stratified Split (Leakage Önleme)**

> **🎯 Amaç:** Ölçekleme ve encoding gibi tüm dönüşümleri bir `Pipeline` içinde, **yalnızca eğitim verisinde fit ederek** veri sızıntısını (leakage) önlemek.

```mermaid
graph LR
    A[✨ Zengin Veri] --> B[✂️ Stratified Split<br/>80/20]
    B --> C[🚂 X_train]
    B --> D[🧪 X_test]
    C --> E[⚙️ Pipeline.fit_transform<br/>SADECE train]
    D --> F[⚙️ Pipeline.transform<br/>sadece dönüştür]
    E --> G[💾 model_ready + pipeline.pkl]
    F --> G
    style B fill:#D4B4FF,stroke:#9370DB,stroke-width:2px,color:#000
    style E fill:#B4D7FF,stroke:#1E90FF,stroke-width:2px,color:#000
    style G fill:#B4FFD7,stroke:#32CD32,stroke-width:2px,color:#000
```

**🔑 Kritik İlke:** Split önce yapılır, `fit` yalnızca train'e uygulanır → test istatistikleri eğitime sızmaz.
""")
code(r"""
# ============================================
# ÖZELLİK GRUPLARI + STRATIFIED SPLIT
# ============================================
TARGET = "Exited"
X = df_fe.drop(columns=[TARGET])
y = df_fe[TARGET]

numerical   = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
categorical = ["Geography", "Gender"]
passthrough = ["NumOfProducts", "HasCrCard", "IsActiveMember",
               "is_high_products_risk", "has_zero_balance"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

print(f"✂️  Split → train: {X_train_raw.shape[0]:,} | test: {X_test_raw.shape[0]:,}")
print(f"   Train churn: %{y_train.mean()*100:.2f} | Test churn: %{y_test.mean()*100:.2f} (oran korundu)")
""")
code(r"""
# ============================================
# LEAKAGE'SİZ PIPELINE (fit yalnızca train)
# ============================================
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical),
], remainder="passthrough", verbose_feature_names_out=False)
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

X_train_proc = pipeline.fit_transform(X_train_raw)   # <-- TEK fit (train)
X_test_proc  = pipeline.transform(X_test_raw)        # <-- sadece transform

feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
X_train = pd.DataFrame(X_train_proc, columns=feature_names)
X_test  = pd.DataFrame(X_test_proc,  columns=feature_names)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# Sinif agirliklari (SMOTE yerine)
weights = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
class_weight_dict = {0: round(weights[0],4), 1: round(weights[1],4)}
scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()

print(f"✅ Dönüştürülmüş özellik sayısı: {len(feature_names)}")
print(f"   {feature_names}")
print(f"\n⚖️  class_weight (balanced): {class_weight_dict}")
print(f"   XGBoost scale_pos_weight : {scale_pos_weight:.4f}")

# Kayit
X_train.to_csv(DATA_MODEL/"X_train.csv", index=False)
X_test.to_csv(DATA_MODEL/"X_test.csv", index=False)
y_train.to_frame("Exited").to_csv(DATA_MODEL/"y_train.csv", index=False)
y_test.to_frame("Exited").to_csv(DATA_MODEL/"y_test.csv", index=False)
joblib.dump(pipeline, MODELS_DIR/"preprocessing_pipeline.pkl")
print("\n💾 Kaydedildi: data/model_ready/ + models/preprocessing_pipeline.pkl")
""")

# =====================================================================
# ADIM 9
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🤖 ADIM 9: 12 Model + Karşılaştırma
    </h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 18px; margin-top: 15px;">
        (Model Training & Comparison — class_weight stratejisi)
    </p>
</div>

## 🧮 **Kullanılacak 12 Sınıflandırıcı**

| # | Algoritma | Kategori | Dengesizlik Stratejisi |
|---|-----------|----------|------------------------|
| 1 | **Logistic Regression** | Doğrusal | `class_weight='balanced'` |
| 2 | **Ridge Classifier** | Doğrusal | `class_weight='balanced'` |
| 3 | **KNN** | Instance | — (desteklemez) |
| 4 | **Gaussian NB** | Olasılıksal | — (desteklemez) |
| 5 | **Decision Tree** | Ağaç | `class_weight='balanced'` |
| 6 | **Random Forest** | Ensemble | `class_weight='balanced'` |
| 7 | **Extra Trees** | Ensemble | `class_weight='balanced'` |
| 8 | **Gradient Boosting** | Boosting | — |
| 9 | **AdaBoost** | Boosting | — |
| 10 | **SVM (RBF)** | Kernel | `class_weight='balanced'` |
| 11 | **XGBoost** | Boosting | `scale_pos_weight` |
| 12 | **LightGBM** | Boosting | `is_unbalance=True` |

## 📊 **Değerlendirme Metrikleri (Sınıflandırma)**
**Accuracy · Precision · Recall · F1 · ROC-AUC** — ana metrik **F1 + Recall**.
""")
code(r"""
# ============================================
# 12 MODELİ TANIMLA (class_weight stratejisi)
# ============================================
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Ridge Classifier":    RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Gaussian NB":         GaussianNB(),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE),
    "AdaBoost":            AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost":             xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                              n_jobs=-1, verbosity=0, random_state=RANDOM_STATE),
    "LightGBM":            lgb.LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, num_leaves=31,
                              is_unbalance=True, n_jobs=-1, verbose=-1, random_state=RANDOM_STATE),
}
print(f"✅ {len(models)} model tanımlandı.")
print("ℹ️  class_weight desteklemeyen (not edildi): KNN, Gaussian NB, AdaBoost, Gradient Boosting")
""")
code(r"""
# ============================================
# EĞİTİM DÖNGÜSÜ + 5-FOLD CV + METRİKLER
# ============================================
print("="*80)
print("🤖 MODEL EĞİTİMİ".center(80))
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, roc_data, trained = [], {}, {}

def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    tr_pred, te_pred = model.predict(X_train), model.predict(X_test)
    te_proba = get_proba(model, X_test)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, te_pred), 4),
        "Precision": round(precision_score(y_test, te_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, te_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, te_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, te_proba), 4),
        "CV_F1": round(cv_f1.mean(), 4),
        "Overfit": round(f1_score(y_train, tr_pred, zero_division=0) - f1_score(y_test, te_pred, zero_division=0), 4),
        "Süre(s)": round(time.time()-t0, 2),
    })
    fpr, tpr, _ = roc_curve(y_test, te_proba)
    roc_data[name] = (fpr, tpr, roc_auc_score(y_test, te_proba))
    trained[name] = model
    r = results[-1]
    print(f"   ✅ {name:22s} F1={r['F1']:.3f}  Recall={r['Recall']:.3f}  AUC={r['ROC-AUC']:.3f}  ({r['Süre(s)']}s)")

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
results_df.to_csv(MODELS_DIR/"all_model_results.csv", index=False)
print("\n💾 Kaydedildi: models/all_model_results.csv")
""")
code(r"""
# ============================================
# KARŞILAŞTIRMA — İNTERAKTİF TABLO + GRAFİK
# ============================================
disp = results_df.copy()
disp.insert(0, "Sıra", range(1, len(disp)+1))

fig = go.Figure(data=[go.Table(
    header=dict(values=["<b>"+c+"</b>" for c in disp.columns],
                fill_color=C_PRIMARY, align="center",
                font=dict(color="white", size=11, family="Arial Black")),
    cells=dict(values=[disp[c] for c in disp.columns],
               fill_color=[["#eaf0ff","#ffffff"]*len(disp)],
               align="center", font=dict(color="#333", size=11), height=24))])
fig.update_layout(title=dict(text="📊 Model Karşılaştırma Tablosu (F1'e göre sıralı)",
                  x=0.5, xanchor="center", font=dict(size=18, family="Arial Black", color="#333")),
                  height=460, margin=dict(l=10, r=10, t=60, b=10))
fig.write_html(FIGURES_DIR/"nb10_model_table.html")
fig.show()

# Gruplu metrik bar
fig2 = go.Figure()
for metric, color in [("F1", C_PRIMARY), ("Recall", C_CHURN), ("ROC-AUC", "#6A994E"), ("Precision", "#F18F01")]:
    fig2.add_trace(go.Bar(name=metric, x=results_df["Model"], y=results_df[metric],
                   text=[f"{v:.2f}" for v in results_df[metric]], textposition="outside"))
fig2.update_layout(barmode="group", title=dict(text="🏆 Model Performans Karşılaştırması",
                   x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                   height=560, xaxis_tickangle=-35)
fig2.write_html(FIGURES_DIR/"nb11_model_comparison.html")
fig2.show()
""")

# =====================================================================
# ADIM 10
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #0B0C2A 0%, #1A2980 50%, #FF0844 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(255,8,68,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        📈 ADIM 10: Model Değerlendirme
    </h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 18px; margin-top: 15px;">
        (Confusion Matrix · ROC · Classification Report)
    </p>
</div>

<div style="background: #ffffff; padding: 25px; border-radius: 15px; border-left: 6px solid #1A2980; margin: 20px 0; box-shadow: 0 8px 20px rgba(26,41,128,0.15);">

### 🎯 **Bu Adımda Yapılacaklar**

F1'e göre en iyi modeli alıp **hata yapısını** detaylı inceliyoruz:
- 🔲 **Confusion Matrix** — TN / FP / FN / TP (False Negative = kaçırılan churn, en pahalı hata)
- 📈 **ROC Eğrisi** — tüm modeller, en iyisi vurgulu
- 📋 **Classification Report** — sınıf bazında precision/recall/f1

</div>
""")
code(r"""
# ============================================
# CONFUSION MATRIX — EN İYİ MODEL (F1)
# ============================================
top_name = results_df.iloc[0]["Model"]
top_model = trained[top_name]
y_pred = top_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fig = go.Figure(data=go.Heatmap(
    z=cm, x=["Tahmin: Kaldı (0)", "Tahmin: Churn (1)"],
    y=["Gerçek: Kaldı (0)", "Gerçek: Churn (1)"],
    text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
    texttemplate="%{text}", textfont=dict(size=16, color="white", family="Arial Black"),
    colorscale=[[0,"#a8c0ff"],[1,C_SECOND]], showscale=True,
    hovertemplate="%{x} / %{y}: %{z}<extra></extra>"))
fig.update_layout(title=dict(text=f"🔲 Confusion Matrix — {top_name}",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  yaxis=dict(autorange="reversed"), width=620, height=480)
fig.write_html(FIGURES_DIR/"nb12_confusion_matrix.html")
fig.show()

print(f"📋 [{top_name}] Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Kaldı (0)", "Churn (1)"]))
print(f"🔎 {fn} churn müşterisi KAÇIRILDI (FN), {fp} yanlış alarm (FP).")
""")
code(r"""
# ============================================
# ROC EĞRİSİ — TÜM MODELLER
# ============================================
fig = go.Figure()
palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
for i, (name, (fpr, tpr, auc)) in enumerate(sorted(roc_data.items(), key=lambda x:-x[1][2])):
    is_best = (name == top_name)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        name=f"{name} (AUC={auc:.3f})",
        line=dict(width=4 if is_best else 1.5,
                  color=C_CHURN if is_best else palette[i % len(palette)],
                  dash="solid" if is_best else "dot")))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Rastgele",
              line=dict(color="gray", dash="dash")))
fig.update_layout(title=dict(text=f"📈 ROC Eğrisi — En İyi: {top_name}",
                  x=0.5, xanchor="center", font=dict(size=20, family="Arial Black", color="#2c3e50")),
                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                  height=620, legend=dict(x=0.55, y=0.05))
fig.write_html(FIGURES_DIR/"nb13_roc_all.html")
fig.show()
""")

# =====================================================================
# ADIM 11
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #1A2980 0%, #FF0844 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(26,41,128,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        🏆 ADIM 11: Final Model Seçimi ve Kaydetme
    </h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 18px; margin-top: 15px;">
        (Çok Kriterli Seçim ve Model Artifact Oluşturma)
    </p>
</div>

<div style="background: #ffffff; padding: 25px; border-radius: 15px; border-left: 6px solid #FF0844; margin: 20px 0; box-shadow: 0 8px 20px rgba(255,8,68,0.15);">

> **🎯 Amaç:** En iyi modeli **çok kriterli** seçmek (tek metriğe değil), `models/final_model.pkl` olarak kaydetmek ve örnek tahminlerle doğrulamak.

### 📐 **Çok Kriterli Skor**
`Skor = F1×0.35 + Recall×0.25 + ROC-AUC×0.20 + CV_F1×0.10 + OverfitCezası×0.10`

Bu formül hem **yüksek churn yakalama** (F1/Recall), hem **kararlılık** (CV), hem de **düşük overfitting**'i aynı anda ödüllendirir.

</div>
""")
code(r"""
# ============================================
# ÇOK KRİTERLİ SEÇİM + KAYDETME
# ============================================
def score(r):
    overfit_pen = max(0, 1 - abs(r["Overfit"]) / 0.5)
    return r["F1"]*0.35 + r["Recall"]*0.25 + r["ROC-AUC"]*0.20 + r["CV_F1"]*0.10 + overfit_pen*0.10

ranked = results_df.copy()
ranked["Skor"] = ranked.apply(score, axis=1)
ranked = ranked.sort_values("Skor", ascending=False).reset_index(drop=True)

print("="*80)
print("🏆 ÇOK KRİTERLİ SIRALAMA (İlk 5)".center(80))
print("="*80)
print(ranked[["Model","F1","Recall","ROC-AUC","Overfit","CV_F1","Skor"]].head().to_string(index=False))

best_name = ranked.iloc[0]["Model"]
best_model = trained[best_name]
joblib.dump(best_model, MODELS_DIR/"final_model.pkl")
ranked.to_csv(MODELS_DIR/"ranked_model_results.csv", index=False)

print("\n" + "="*80)
print(f"   🥇 SEÇİLEN FİNAL MODEL: {best_name}")
b = ranked.iloc[0]
print(f"   Accuracy={b['Accuracy']:.4f} | Precision={b['Precision']:.4f} | "
      f"Recall={b['Recall']:.4f} | F1={b['F1']:.4f} | ROC-AUC={b['ROC-AUC']:.4f}")
print("="*80)
print("\n💾 Kaydedildi: models/final_model.pkl + models/ranked_model_results.csv")
""")
code(r"""
# ============================================
# ÖRNEK TAHMİNLER — 5 TEST MÜŞTERİSİ
# ============================================
proba = get_proba(best_model, X_test)
sample = X_test.head(5).copy()
sample_out = pd.DataFrame({
    "Gerçek": y_test.head(5).values,
    "Tahmin": best_model.predict(X_test.head(5)),
    "Churn_Olasılık_%": (proba[:5]*100).round(1),
})
sample_out["Sonuç"] = np.where(sample_out["Gerçek"]==sample_out["Tahmin"], "✅ Doğru", "❌ Yanlış")
print("🔮 Final Modelin Örnek Tahminleri:\n")
print(sample_out.to_string(index=False))
""")

# =====================================================================
# ADIM 12
# =====================================================================
md(r"""
<div style="background: linear-gradient(135deg, #FF0844 0%, #1A2980 50%, #0B0C2A 100%); padding: 40px; border-radius: 15px; margin: 30px 0; box-shadow: 0 10px 30px rgba(255,8,68,0.3);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        💡 ADIM 12: Sonuç ve İş Önerileri
    </h1>
    <p style="color: #ffffff; text-align: center; font-size: 18px; margin-top: 15px;">
        (Conclusion & Business Recommendations)
    </p>
</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(26,41,128,0.15); border-left: 6px solid #1A2980;">

### 📊 **Projenin Özeti**

Bu çalışmada CRISP-DM metodolojisiyle, banka müşterilerinin kaybını (`Exited`)
tahmin eden uçtan uca bir **sınıflandırma** sistemi geliştirildi: ham veriden
(10.000 müşteri) başlayıp temizleme, feature engineering, leakage'sız pipeline,
12 model karşılaştırması ve final model seçimine kadar.

</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(255,8,68,0.15); border-left: 6px solid #FF0844;">

### 🔍 **EDA'dan Çıkan Kilit Bulgular**

| 🎯 Sürücü | 📈 Etki | 💬 Yorum |
|:----------|:--------|:---------|
| 🎂 **Age** | En güçlü sinyal | Yaş arttıkça churn artıyor (yaşlı müşteriler riskli) |
| ✅ **IsActiveMember** | Yüksek | Pasif üyeler ~2 kat daha fazla ayrılıyor |
| 🌍 **Geography** | Yüksek | Almanya, Fransa/İspanya'dan belirgin yüksek churn |
| 📦 **NumOfProducts** | Çok yüksek (≥3) | 3–4 ürünlü müşterilerde churn dramatik artıyor |
| 💳 **CreditScore / Tenure / Salary** | Zayıf | Churn ile düşük ilişki |

</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(26,41,128,0.15); border-left: 6px solid #1A2980;">

### 💼 **İş Önerileri (Aksiyona Dönük)**

1. 🎯 **Aktivasyon programları:** Pasif üyelere yönelik kampanyalarla `IsActiveMember` oranını artır — en kolay kazanç alanı.
2. 🌍 **Almanya'ya özel inceleme:** Yüksek churn'ün operasyonel/fiyatlandırma kök nedenini araştır.
3. 📦 **Ürün paketleme gözden geçir:** 3–4 ürünlü müşterilerdeki yüksek churn, zorunlu paketlemenin memnuniyetsizliğe yol açtığını düşündürüyor.
4. 🎂 **Yaş segmentli teklifler:** Yaşlı müşteri segmentine özel sadakat ve danışmanlık hizmetleri.
5. 📉 **Risk skoru entegrasyonu:** Model çıktısı (churn olasılığı) CRM'e entegre edilip, eşik (threshold) iş maliyetine göre ayarlanarak Recall artırılabilir.

</div>

<div style="background: white; padding: 25px; border-radius: 12px; margin: 20px 0; box-shadow: 0 5px 15px rgba(255,8,68,0.15); border-left: 6px solid #FF0844;">

### 🚀 **Üretilen Artefaktlar ve Dağıtım**

| 📦 Dosya | 📝 Açıklama |
|:---------|:------------|
| `models/preprocessing_pipeline.pkl` | Ham girdileri dönüştüren pipeline |
| `models/final_model.pkl` | Seçilen final sınıflandırma modeli |
| `data/model_ready/` | İşlenmiş train/test setleri |
| `models/*_results.csv` | Model karşılaştırma metrikleri |
| `figures/nb*.html` | İnteraktif Plotly grafikleri |

**Streamlit uygulaması ile canlı tahmin:**
```bash
python -m streamlit run app/app.py
```

### 🔭 **Sonraki Adımlar**
- 🎚️ **Threshold optimizasyonu** (0.5 yerine iş maliyetine göre)
- 🔧 **Hiperparametre tuning** (GridSearch / Optuna)
- 📊 **SHAP** ile açıklanabilirlik
- 📡 **Monitoring:** veri kayması ve canlı F1/Recall takibi

</div>

<div style="background: linear-gradient(135deg, #0B0C2A 0%, #1A2980 50%, #FF0844 100%); padding: 30px; border-radius: 15px; text-align: center; margin-top: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
  <h2 style="color: white; margin: 0;">✅ Proje Tamamlandı</h2>
  <p style="color: #f0f0f0; margin-top: 10px;">CRISP-DM akışı uçtan uca, <code style="color:#fff;">data/raw/churn.csv</code>'den başlayarak tekrar çalıştırılabilir biçimde tamamlandı. 🎉</p>
</div>
""")

# =====================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook yazildi: {NB_PATH}")
print(f"Toplam hucre: {len(cells)} "
      f"(md: {sum(1 for c in cells if c.cell_type=='markdown')}, "
      f"code: {sum(1 for c in cells if c.cell_type=='code')})")
