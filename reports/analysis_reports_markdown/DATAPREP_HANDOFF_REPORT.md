# MODEL EXPERT HANDOFF RAPORU
**Tarih:** 2026-05-29 23:37
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

## Ozellik Listesi (13 Ozellik)

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
- fit() YALNIZCA X_train (8000 ornek) uzerinde yapildi.
- X_test (2000 ornek) yalnizca transform() ile islendi.
- sklearn Pipeline + ColumnTransformer kullanildi.

---

## Sinif Dengesizligi Stratejisi

**SMOTE UYGULANMADI.** class_weight='balanced' onerilir.

| Sinif | Ornek Sayisi | Oran | class_weight |
|-------|-------------|------|--------------|
| 0 (Kaldi)    | 7973 | %79.7 | 0.6271 |
| 1 (Ayrildi)  | 2027 | %20.3 | 2.4667 |

**Model parametreleri:**
- RandomForest / LogisticRegression: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=3.93`
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
| Train | 8000 | %20.28 |
| Test  | 2000 | %20.25 |

stratify=y, random_state=42, test_size=0.20

---

## Leakage Durumu

**YOK (Dusuk Risk)**

- Tum donusumler Pipeline icinde; test seti fit edilmedi
- Hedef degiskeni kopyalayan ozellik bulunamadi (max |r|=0.285)
- Temporal leakage: Veri setsinde zaman boyutu yok

---

## Onerilen Model Turleri

1. **RandomForestClassifier** — class_weight='balanced', n_estimators>=200
2. **XGBClassifier** — scale_pos_weight=3.93
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
    X_train.csv   (8000 x 13)
    X_test.csv    (2000 x 13)
    y_train.csv                     (8000 x 1)
    y_test.csv                      (2000 x 1)
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
