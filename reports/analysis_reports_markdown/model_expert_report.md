# Model Expert — Churn Binary Classification Raporu
Olusturulma Tarihi: 2026-05-29

---

## PHASE 1: Veri Dogrulama

| Boyut | Deger |
|---|---|
| X_train | 8000 x 13 |
| X_test  | 2000 x 13 |
| y_train sinif 0 | 6378 (%79.7) |
| y_train sinif 1 | 1622 (%20.3) |
| y_test sinif 0  | 1595 (%79.8) |
| y_test sinif 1  | 405 (%20.2) |
| Eksik deger     | 0 |

---

## PHASE 2: Problem Tanimlama

- **Tip:** Binary Classification
- **Hedef:** Exited (musteri churn etmis mi?)
- **Dengesizlik:** ~20% pozitif sinif, oran ~4:1
- **Strateji:** SMOTE YOK — class_weight='balanced' (destekleyen modellerde)
- **Ana Metrik:** F1 (pozitif sinif) + Recall (oncelikli)

---

## PHASE 5: Kurulan Modeller

Toplam 16 model denendi.

class_weight='balanced' DESTEKLEYENLER:
LogisticRegression, RidgeClassifier, DecisionTree, RandomForest, ExtraTrees, SVM (RBF)

class_weight DESTEKLEMEYEN (not edildi):
KNN, GaussianNB, Bagging, GradientBoosting, AdaBoost, MLP, XGBoost, LightGBM, CatBoost

---

## PHASE 6: Model Egitim Parametreleri

| Model | n_estimators | Ozel Parametre |
|---|---|---|
| Logistic Regression | - | class_weight='balanced', max_iter=1000 |
| Random Forest | 300 | class_weight='balanced', n_jobs=-1 |
| Extra Trees | 300 | class_weight='balanced', n_jobs=-1 |
| Gradient Boosting | 200 | lr=0.1, max_depth=4 |
| XGBoost | 300 | scale_pos_weight=3.93, lr=0.05 |
| LightGBM | 300 | is_unbalance=True, lr=0.05 |
| CatBoost | 300 | auto_class_weights='Balanced', lr=0.05 |
| SVM (RBF) | - | class_weight='balanced', probability=True |
| MLP | - | hidden=(128,64,32), early_stopping=True |

---

## PHASE 9: Final Model Secimi

### Secilen Model: Logistic Regression

| Metrik | Deger |
|---|---|
| Test F1 (pozitif sinif) | 0.4866 |
| Recall | 0.6716 |
| Precision | 0.3815 |
| ROC-AUC | 0.7629 |
| Accuracy | 0.7130 |
| Overfit Gap (Train-Test F1) | +0.0103 |
| CV F1 Ortalama | 0.4932 |
| CV F1 Std | 0.0246 |
| Cok Kriterli Puan | 0.6381 |

**Secim Gerekce:** Final model olarak Logistic Regression secilmistir. Bu secim yalnizca en yuksek
test F1 skoruna degil; CV kararliligi, train-test overfitting farki, Recall performansi
ve uretimde uygulanabilirlik kriterlerine dayanir. Sinif dengesizligi nedeniyle
tek basina accuracy kullanilmamis; F1 ve Recall on plana cikarilmistir.

---

## PHASE 10: Hata Analizi

| | Tahmin: 0 | Tahmin: 1 |
|---|---|---|
| Gercek: 0 | TN = 1154 | FP = 441 |
| Gercek: 1 | FN = 133 | TP = 272 |

- **133 False Negative:** Churn edecek musterilerin kacirilan tahminleri — is maliyeti en yuksek hata.
- **441 False Positive:** Yanlis alarm — gereksiz retention kampanyasi maliyeti.
- Recall = 0.6716: Churn eden musterilerin %67.2'i dogru yakalanmaktadir.

---

## Gorsel Ciktilar

| Dosya | Aciklama |
|---|---|
| figures/model_phase7_performance_comparison | 12+ Model F1/Recall/AUC karsilastirmasi |
| figures/model_phase7_cv_stability | CV kararliligi |
| figures/model_phase7_overfitting_analysis | Train vs Test overfitting |
| figures/model_phase7_training_time | Egitim suresi analizi |
| figures/model_phase7_leadership_matrix | Cok kriterli liderlik matrisi |
| figures/model_phase10_final_confusion_matrix | Final model confusion matrix |
| figures/model_phase10_roc_all_models | Tum modeller ROC karsilastirmasi |
| figures/model_phase10_precision_recall_curve | PR egrisi |

---

## Sonraki Adim — Explainability Expert

- **Final Model:** Logistic Regression
- **Model Dosyasi:** models/final_model.pkl
- **Aciklanabilirlik Onerisi:** SHAP feature importance + permutation importance
- **Kritik Featurelar:** Age, Balance, NumOfProducts, IsActiveMember, Geography_Germany
- **Dikkat:** Azinlik sinifinda yuksek Recall saglandigi teyit edildi

---

## Sonraki Adim — Deployment Expert

- **Model Dosyasi:** models/final_model.pkl
- **Pipeline:** models/preprocessing_pipeline.pkl + final_model.pkl
- **Input Schema:** CreditScore, Age, Tenure, Balance, EstimatedSalary, Geography_Germany,
  Geography_Spain, Gender_Male, NumOfProducts, HasCrCard, IsActiveMember,
  is_high_products_risk, has_zero_balance
- **Output:** Tahmin sinifi (0/1) + olasilik
- **Monitoring:** Data drift, class distribution drift, F1 dusus izlenmeli
- **Risk:** Azinlik sinif (Exited=1) — threshold optimizasyonu one cikarilabilir
