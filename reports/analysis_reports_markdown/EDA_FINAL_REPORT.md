# Keşifsel Veri Analizi Raporu
## Banka Müşteri Kaybı (Churn) Tahmini

**Tarih:** 2026-05-29
**Metodoloji:** CRISP-DM — Data Understanding Aşaması
**Analiz Aracı:** EDA Expert (Agentik 7-Aşamalı Pipeline)
**Hedef Değişken:** `Exited` (0 = Müşteri kaldı, 1 = Müşteri ayrıldı)

---

## 1. Yönetici Özeti

Bu rapor, 10.000 banka müşterisini kapsayan churn veri seti üzerinde gerçekleştirilen kapsamlı keşifsel veri analizinin bulgularını sunar. Veri seti yüksek kaliteli yapısıyla öne çıkmaktadır: eksik değer bulunmamakta, mükerrer kayıt yer almamakta ve mantıksal tutarsızlık tespit edilmemektedir.

**Temel fırsatlar:** Yaş ve aktif üyelik durumu en güçlü churn öngörücüleridir. 45 yaş üstü pasif Alman müşteriler en yüksek risk segmentini oluşturmaktadır.

**Temel riskler:** Hedef değişkende %79.7 / %20.3 sınıf dengesizliği mevcuttur. Kategorik değişkenlerin encode edilmesi ve sayısal değişkenlerin ölçeklenmesi gerekmektedir. Bu adımlar tamamlanmadan modellemeye geçilmesi önerilmez.

**Model hazırlık kararı: KISMEN HAZIR**

---

## 2. Veri Setinin Genel Profili

| Özellik | Değer |
|---|---|
| Satır sayısı | 10.000 |
| Sütun sayısı | 14 |
| Toplam hücre | 140.000 |
| Eksik değer | **0** |
| Mükerrer kayıt | **0** |
| Hedef değişken | `Exited` (ikili sınıflandırma) |
| Bellek kullanımı | ~2.5 MB |

### Sütun Yapısı

| Sütun | Tip | Eşsiz Değer | Rol |
|---|---|---|---|
| RowNumber | int64 | 10.000 | ID — modelden çıkar |
| CustomerId | int64 | 9.805 | ID — modelden çıkar |
| Surname | str | 40 | Gereksiz — modelden çıkar |
| CreditScore | int64 | 466 | Sayısal özellik |
| Geography | str | 3 | Kategorik özellik |
| Gender | str | 2 | Kategorik özellik |
| Age | int64 | 72 | Sayısal özellik — En güçlü öngörücü |
| Tenure | int64 | 11 | Sayısal özellik |
| Balance | float64 | 6.457 | Sayısal özellik |
| NumOfProducts | int64 | 4 | Diskret özellik |
| HasCrCard | int64 | 2 | İkili özellik |
| IsActiveMember | int64 | 2 | İkili özellik — Güçlü öngörücü |
| EstimatedSalary | float64 | 9.998 | Sayısal özellik |
| Exited | int64 | 2 | **Hedef değişken** |

---

## 3. Kritik Teknik Bulgular

### PHASE 1: Veri Genel Bakış

**Yapılan Analiz:** Veri seti yüklendi, yapısal özellikler, veri tipleri, boşluk analizi ve ilk istatistiksel özet çıkarıldı.

**Bulgular:**
- Veri seti tam ve eksiksizdir; 10.000 satır, 14 sütun
- `RowNumber`, `CustomerId`, `Surname` sütunları modelleme için anlamsız tanımlayıcı alanlar
- `Surname` sütununda yalnızca 40 eşsiz değer mevcuttur; içerik bilgi taşımaz
- 195 adet `CustomerId` tekrarı tespit edildi — aynı müşterinin birden fazla kayıt içerip içermediği araştırılmalıdır

**Risk:** CustomerId'de 195 tekrar, veri toplama sürecindeki olası duplikasyona işaret edebilir.

**Kaydedilen Görseller:**
- `figures/phase1_column_overview.html`

---

### PHASE 2: Tek Değişkenli Analiz

**Yapılan Analiz:** Tüm sayısal değişkenler için histogram, boxplot, çarpıklık ve kurtosis hesaplandı. Kategorik değişkenler için frekans ve oran tabloları oluşturuldu.

**Bulgular — Sayısal Değişkenler:**

| Değişken | Ortalama | Medyan | Çarpıklık | Outlier % |
|---|---|---|---|---|
| CreditScore | 650.8 | 650.0 | -0.077 | 0.29% |
| Age | 38.8 | 37.0 | 0.650 | 1.27% |
| Tenure | 5.0 | 5.0 | -0.016 | 0.00% |
| Balance | 77.353 | 97.216 | -0.169 | 0.00% |
| EstimatedSalary | 100.260 | 100.251 | -0.014 | 0.00% |

**Önemli Bulgular:**
- Tüm sayısal değişkenlerde çarpıklık düşük (|skewness| < 1) — dönüşüm gerekmez
- `Age` hafif sağa çarpık; 127 outlier (%1.27) — düşük risk
- `Balance`'ın %35.4'ü sıfır değer içeriyor — bu müşteriler özel bir davranış sergiliyor olabilir
- `Tenure` ve `EstimatedSalary` neredeyse mükemmel düzgün dağılımlı

**Bulgular — Kategorik Değişkenler:**

| Değişken | Kategoriler | Dağılım |
|---|---|---|
| Geography | France, Spain, Germany | %49.8 / %25.6 / %24.6 |
| Gender | Male, Female | %54.4 / %45.6 |

**Bulgular — Diskret/İkili Değişkenler:**
- `NumOfProducts`: %69.1 tek ürün, %30.4 iki ürün, %0.5 üç ürün
- `HasCrCard`: %71.6 kredi kartı sahibi
- `IsActiveMember`: %51.5 aktif üye

**Hedef Değişken Dağılımı:**
- Exited=0 (Kaldı):   7.973 müşteri (%79.7)
- Exited=1 (Ayrıldı): 2.027 müşteri (%20.3)
- **Sınıf dengesizliği belirgindir.**

**Kaydedilen Görseller:**
- `figures/phase2_histograms_continuous.png`
- `figures/phase2_boxplots_continuous.png`
- `figures/phase2_bar_geography.png`
- `figures/phase2_pie_gender.png`
- `figures/phase2_bar_discrete_binary.png`
- `figures/phase2_target_exited_distribution.png`

---

### PHASE 3: İki Değişkenli Analiz (Exited ile İlişkiler)

**Yapılan Analiz:** Her özelliğin hedef değişken `Exited` ile istatistiksel ilişkisi Mann-Whitney U testi, churn oranı hesabı ve görselleştirmelerle incelendi.

**Sayısal Değişkenler vs Exited — İstatistiksel Bulgular:**

| Değişken | Kaldı Ort. | Ayrıldı Ort. | Fark | p-değeri | Anlamlılık |
|---|---|---|---|---|---|
| **Age** | 37.1 | 45.6 | **+23.1%** | <0.001 | *** |
| **Balance** | 76.180 | 81.967 | +7.6% | 0.0003 | *** |
| CreditScore | 651.4 | 648.2 | -0.5% | 0.169 | ns |
| Tenure | 5.04 | 5.00 | -0.7% | 0.671 | ns |
| EstimatedSalary | 100.577 | 99.011 | -1.6% | 0.274 | ns |

**Kritik Bulgu:** `Age` istatistiksel olarak en güçlü ayrışım yaratan değişkendir. Ayrılan müşterilerin ortalama yaşı kalan müşterilere kıyasla %23 daha yüksektir.

**`CreditScore`, `Tenure` ve `EstimatedSalary` ile churn arasında istatistiksel olarak anlamlı ilişki bulunmamaktadır.**

**Kategorik Değişkenler vs Exited:**

| Değişken | Kategori | Churn Oranı |
|---|---|---|
| Geography | Germany | **%28.8** |
| Geography | France | %17.7 |
| Geography | Spain | %17.2 |
| Gender | Female | **%22.9** |
| Gender | Male | %18.1 |

**Almanya segmenti Fransa ve İspanya'ya kıyasla 1.6 kat daha yüksek churn oranına sahiptir.**

**Diskret Değişkenler vs Exited:**

| NumOfProducts | Churn Oranı |
|---|---|
| 1 | %19.5 |
| 2 | %17.9 |
| **3** | **%63.6** |
| **4** | **%66.0** |

**Kritik anomali:** 3 veya 4 ürün sahibi müşteriler dramatik biçimde yüksek churn sergilmektedir. Bu doğrusal olmayan ilişki, özel bir kural/bayrak değişkeni gerektirmektedir.

**IsActiveMember vs Churn:**
- Aktif üye: %13.4 churn
- Pasif üye: **%27.5 churn** — neredeyse 2 kat fark

**Balance Grubu vs Churn:**
- Sıfır bakiye: %18.1
- Pozitif bakiye: %21.5

**Kaydedilen Görseller:**
- `figures/phase3_boxplot_numeric_vs_exited.png`
- `figures/phase3_histogram_age_vs_exited.png`
- `figures/phase3_balance_group_churn_rate.png`
- `figures/phase3_categorical_churn_rates.png`
- `figures/phase3_discrete_churn_rates.png`
- `figures/phase3_violin_age_vs_exited.png`
- `figures/phase3_geo_gender_churn.png`

---

### PHASE 4: Çok Değişkenli Analiz

**Yapılan Analiz:** Pearson korelasyon matrisi, pairwise scatter matrix ve ısıl haritalar üretildi. Multicollinearity kontrolü yapıldı.

**Pearson Korelasyon — Exited ile (büyükten küçüğe):**

| Değişken | Korelasyon (r) | Güç |
|---|---|---|
| Age | +0.285 | Orta |
| IsActiveMember | -0.175 | Düşük-Orta |
| Geography (encoded) | +0.102 | Düşük |
| NumOfProducts | +0.088 | Düşük |
| Gender (encoded) | +0.060 | Çok Düşük |
| Balance | +0.037 | Anlamsız |
| HasCrCard | -0.017 | Anlamsız |
| CreditScore | -0.014 | Anlamsız |
| EstimatedSalary | -0.011 | Anlamsız |
| Tenure | -0.004 | Anlamsız |

**Kritik Bulgu: Multicollinearity yoktur.** Hiçbir değişken çifti arasında |r| > 0.50 korelasyon tespit edilmedi. Bu, tüm değişkenlerin modele dahil edilebileceği anlamına gelir.

**Bileşik Segment Analizi (Coğrafya x Aktif Üyelik):**

| Ülke | Pasif Üye Churn | Aktif Üye Churn |
|---|---|---|
| Germany | **%38.4** | %19.2 |
| France | %23.7 | %12.0 |
| Spain | %23.9 | %10.8 |

**Kaydedilen Görseller:**
- `figures/phase4_correlation_matrix.png`
- `figures/phase4_exited_correlations.png`
- `figures/phase4_scatter_matrix.png`
- `figures/phase4_scatter_age_balance_churn.png`
- `figures/phase4_geo_activemember_churn_heatmap.png`

---

### PHASE 5: Veri Kalitesi ve Anomali Tespiti

**Yapılan Analiz:** Eksik değer, mükerrer kayıt, IQR ve Z-score outlier analizi, mantıksal tutarlılık kontrolleri yapıldı.

**Veri Kalitesi Özet Tablosu:**

| Kontrol | Durum | Detay |
|---|---|---|
| Eksik Veri | **Temiz** | Hiçbir değişkende eksik değer yok |
| Mükerrer Kayıt (Tam Satır) | **Temiz** | Tam tekrar yok |
| CustomerId Tekrarı | Araştır | 195 tekrar var |
| Yaş Sınırı | **Temiz** | Tüm yaşlar [18-92] aralığında |
| Negatif Balance | **Temiz** | Negatif değer yok |
| CreditScore Sınırı | **Temiz** | Tüm değerler [350-850] |
| NumOfProducts | **Temiz** | Sadece 1-4 değerleri var |

**Outlier Analizi (IQR Yöntemi):**

| Değişken | Outlier Sayısı | Outlier % |
|---|---|---|
| Age | 127 | %1.27 |
| CreditScore | 29 | %0.29 |
| NumOfProducts | 50 | %0.50 |
| Balance | 0 | 0.00% |
| Tenure | 0 | 0.00% |
| EstimatedSalary | 0 | 0.00% |

**Tüm değişkenlerde outlier oranı %5 eşiğinin çok altındadır. Outlier müdahalesi zorunlu değildir.**

**Kritik Yapısal Not — Balance Sıfır Değerleri:**
Müşterilerin %35.4'ünün bakiyesi sıfırdır. Bu bir veri hatası değil, gerçek bir müşteri davranışını yansıtmaktadır. Bu grup için bir bayrak değişkeni (`has_zero_balance`) oluşturulması önerilmektedir.

**Kaydedilen Görseller:**
- `figures/phase5_outlier_ratios_iqr.png`
- `figures/phase5_outlier_scatter_creditscore_age.png`

---

### PHASE 6: İçgörü Üretimi

**Yapılan Analiz:** Yaş grupları, segment analizi ve bileşik değişken etki analizleri yapıldı.

**Yaş Grubu Churn Oranları:**

| Yaş Grubu | Müşteri Sayısı | Churn Oranı |
|---|---|---|
| 18-25 | 1.265 | %8.5 |
| 26-35 | 3.074 | %12.4 |
| 36-45 | 2.987 | %18.8 |
| 46-55 | 1.749 | **%29.8** |
| 56-65 | 635 | **%43.9** |
| 65+ | 290 | **%60.0** |

Yaş ilerledikçe churn oranı monoton biçimde artmaktadır. 45 yaş üstü müşteriler birincil koruma hedefidir.

**En Riskli Segmentler (Coğrafya x Cinsiyet):**

| Segment | Churn Oranı |
|---|---|
| Almanya / Kadın | **%30.9** |
| Almanya / Erkek | %27.0 |
| Fransa / Kadın | %20.5 |
| İspanya / Kadın | %20.2 |
| Fransa / Erkek | %15.3 |
| İspanya / Erkek | %14.6 |

**İspanyol erkekler en düşük risk grubunu, Alman kadınlar ise en yüksek risk grubunu oluşturmaktadır.**

**Kaydedilen Görseller:**
- `figures/phase6_age_group_churn_rate.png`
- `figures/phase6_segment_churn_rates.png`
- `figures/phase6_numproducts_activemember_churn.png`
- `figures/phase6_feature_impact_ranking.png`

---

## 4. İş Değeri Açısından İçgörüler

### İçgörü 1: Yaş En Güçlü Churn Göstergesidir

**Bulgu:** Churn olan müşterilerin ortalama yaşı 45.6, kalan müşterilerin ise 37.1'dir. 65+ yaş grubunda churn oranı %60'a ulaşmaktadır.

**İş Değeri:** Yaşlı müşteriler için özel sadakat programları ve proaktif iletişim kampanyaları tasarlanmalıdır. 45+ müşteri segmenti öncelikli koruma hedefi olmalıdır.

**Modelleme Etkisi:** `Age` modelin en güçlü özelliği olacaktır (r=0.285). Yaş grubuna dayalı etkileşim değişkenleri (age_group) ek güç sağlayabilir.

---

### İçgörü 2: Pasif Üyeler 2 Kat Daha Fazla Ayrılıyor

**Bulgu:** Aktif üyelerin churn oranı %13.4 iken pasif üyelerde bu oran %27.5'e çıkmaktadır. `IsActiveMember` değişkeni r=-0.175 korelasyonla ikinci güçlü öngörücüdür.

**İş Değeri:** Müşteri aktivasyon programları (mobil bankacılık kullanımı, işlem teşviki) churn'u doğrudan azaltabilir. Pasif müşteri tespiti erken uyarı sistemi kurulabilir.

**Modelleme Etkisi:** Aktivasyon durumu ile yaş etkileşimi (`age x is_active`) güçlü bir türetilmiş özellik olabilir.

---

### İçgörü 3: Almanya Segmenti Sistemik Risk Taşıyor

**Bulgu:** Almanya %28.8 churn oranıyla Fransa (%17.7) ve İspanya (%17.2) karşısında 1.6 kat yüksek churn sergiliyor. Pasif Alman müşterilerde bu oran %38.4'e çıkıyor.

**İş Değeri:** Almanya operasyonunda ürün-pazar uyumunu, müşteri hizmetleri kalitesini ve yerel rekabeti incelemeye almak stratejik öncelik olabilir.

**Modelleme Etkisi:** `Geography` özelliği için target encoding veya interaction feature (`is_germany_passive`) önerilmektedir.

---

### İçgörü 4: 3-4 Ürün Sahibi Müşteriler Kritik Risk Grubu

**Bulgu:** 1-2 ürün sahibi müşterilerde churn %18-20 iken, 3-4 ürün sahibi müşterilerde bu oran %63-66'ya fırlıyor. Bu ilişki doğrusal değildir ve beklenmediktir.

**İş Değeri:** Çok ürün satışının müşteri memnuniyeti ile ilişkisi yeniden değerlendirilmelidir. Zorla ürün bundle'ı memnuniyeti azaltıyor olabilir. Müşteri başına ideal ürün sayısı 1-2 olarak görünmektedir.

**Modelleme Etkisi:** `NumOfProducts` için ikili bayrak değişkeni (`is_high_products: NumOfProducts >= 3`) önerilmektedir.

---

### İçgörü 5: CreditScore, Tenure ve Maaş Churn'u Açıklamıyor

**Bulgu:** `CreditScore`, `Tenure` ve `EstimatedSalary` için Mann-Whitney U testleri istatistiksel anlamsızlık (p > 0.05) göstermektedir.

**İş Değeri:** Müşteri sadakatinin kredi geçmişi, müşteriliğin süresi veya gelir seviyesiyle doğrudan ilişkisi yoktur. Bu, churn'un ağırlıklı olarak demografik (yaş) ve davranışsal (aktif üyelik, ürün sayısı) faktörlerle şekillendiğine işaret etmektedir.

**Modelleme Etkisi:** Bu değişkenler modele dahil edilebilir; ancak düşük öngörü gücü beklenmelidir. Feature selection aşamasında eleme adayı olabilirler.

---

## 5. Data Prep Expert İçin Öneriler

| # | Sorun | Öncelik | Öneri |
|---|---|---|---|
| 1 | ID sütunları: RowNumber, CustomerId, Surname | Yüksek | Modelden çıkar |
| 2 | Sınıf dengesizliği (%79.7 vs %20.3) | Yüksek | SMOTE veya `class_weight='balanced'` |
| 3 | Kategorik encoding: Geography, Gender | Yüksek | One-hot veya target encoding |
| 4 | Feature scaling | Yüksek | StandardScaler (CreditScore, Age, Balance, EstimatedSalary) |
| 5 | Stratified train-test split | Yüksek | `stratify=y`, tipik oran: 80/20 veya 70/30 |
| 6 | NumOfProducts=3-4 anomalisi | Yüksek | `is_high_products_risk` bayrak değişkeni oluştur |
| 7 | Balance=0 yogunluğu (%35.4) | Orta | `has_zero_balance` ikili bayrak değişkeni |
| 8 | CustomerId tekrarı (195 adet) | Orta | Araştır; gerekirse en son kayıt stratejisi uygula |
| 9 | Feature Engineering | Orta | age_group, geo_gender_risk, age_x_active etkileşimi |
| 10 | Outlier işleme | Düşük | Gerekli değil; tüm değişkenlerde oran <%1.3 |

---

## 6. Model Readiness Assessment

### Kontrol Listesi

| Kontrol | Durum | Not |
|---|---|---|
| Eksik veri yönetimi | Geçti | İmputation gerekmez |
| Mükerrer kayıt | Geçti | Satır tekrarı yok |
| Kategorik encoding | Gerekli | Geography + Gender |
| Feature scaling | Gerekli | 4 sayısal değişken |
| Sınıf dengesizliği | Gerekli | SMOTE / class weighting |
| Outlier müdahale | Düşük Risk | Oranlar <%1.5 |
| Veri sızıntısı | Geçti | Risk tespit edilmedi |
| Multicollinearity | Geçti | |r| > 0.50 yok |
| Stratified split | Gerekli | Zorunlu |
| Feature Engineering | Önerilen | Performans artışı beklenir |

### Veri Kalitesi Skorkartı

| Kriter | Puan |
|---|---|
| Eksik Veri | 10/10 |
| Mükerrer Kayıt | 9/10 |
| Outlier Riski | 9/10 |
| Mantık Tutarlılığı | 10/10 |
| Sınıf Dengesi | 5/10 |
| Feature Kalitesi | 7/10 |
| **Genel Hazırlık** | **8/10** |

### Final Karar: KISMEN HAZIR

Veri seti, kalite açısından modelleme için güçlü bir temel sunmaktadır. Ancak aşağıdaki adımlar tamamlanmadan modellemeye geçilmesi kesinlikle önerilmez:

1. ID sütunlarını düşür
2. Geography ve Gender'i encode et
3. Sayısal değişkenleri ölçekle
4. Sınıf dengesizliğini ele al
5. Stratified split uygula

Bu 5 adım tamamlandıktan sonra veri seti **Hazır** statüsüne geçecektir.

---

## 7. Sonuç ve Yol Haritası

### Tamamlanan Aşama
EDA başarıyla tamamlanmıştır. 10 adet içgörü, 5 kritik risk faktörü ve 10 Data Prep önerisi belirlenmiştir.

### Bir Sonraki Adım: Data Preparation

Önerilen sıra:

```
1. RowNumber, CustomerId, Surname sutunlarini duşur
2. Geography -> One-hot encoding (France, Germany, Spain)
3. Gender -> Binary encoding (Female=1, Male=0)
4. Balance=0 -> has_zero_balance flag degiskeni
5. NumOfProducts >= 3 -> is_high_products_risk flag degiskeni
6. StandardScaler: CreditScore, Age, Balance, EstimatedSalary
7. Stratified train-test split (test_size=0.20, stratify=y)
8. SMOTE uygulaması (yalnızca train setine)
```

### Önerilen Model Adayları

| Model | Gerekçe |
|---|---|
| XGBoost / LightGBM | Sınıf dengesizliği için güçlü, doğrusal olmayan ilişkileri yakalar |
| Random Forest | Yorumlanabilir, robust |
| Logistic Regression | Baseline karşılaştırması için |

### Önerilen Değerlendirme Metrikleri

- **Primary:** F1-Score (sınıf dengesizliği nedeniyle Accuracy kullanılmamalı)
- **Secondary:** ROC-AUC, Precision-Recall AUC
- **Business:** Recall (churn olan müşteriyi kaçırmama öncelikli ise)

---

## 8. Üretilen Dosyalar

### Görseller (figures/)

| Dosya | Açıklama |
|---|---|
| phase1_column_overview | Sütun yapısı genel bakış |
| phase2_histograms_continuous | Sürekli değişken histogramları |
| phase2_boxplots_continuous | Sürekli değişken boxplot'ları |
| phase2_bar_geography | Coğrafya dağılımı |
| phase2_pie_gender | Cinsiyet dağılımı |
| phase2_bar_discrete_binary | Diskret/ikili değişken dağılımları |
| phase2_target_exited_distribution | Hedef değişken dağılımı |
| phase3_boxplot_numeric_vs_exited | Sayısal vs churn boxplot |
| phase3_histogram_age_vs_exited | Yaş histogramı churn karşılaştırması |
| phase3_balance_group_churn_rate | Balance grubu churn oranı |
| phase3_categorical_churn_rates | Kategorik değişken churn oranları |
| phase3_discrete_churn_rates | Diskret değişken churn oranları |
| phase3_violin_age_vs_exited | Yaş violin plot |
| phase3_geo_gender_churn | Coğrafya x cinsiyet churn |
| phase4_correlation_matrix | Pearson korelasyon matrisi |
| phase4_exited_correlations | Exited ile korelasyon sıralaması |
| phase4_scatter_matrix | Scatter matrix |
| phase4_scatter_age_balance_churn | Yaş x bakiye scatter |
| phase4_geo_activemember_churn_heatmap | Coğrafya x aktif üye ısıl harita |
| phase5_outlier_ratios_iqr | Outlier oranları |
| phase5_outlier_scatter_creditscore_age | Outlier tespiti scatter |
| phase6_age_group_churn_rate | Yaş grubu churn oranları |
| phase6_segment_churn_rates | Segment churn oranları |
| phase6_numproducts_activemember_churn | Ürün x üyelik churn |
| phase6_feature_impact_ranking | Feature etki sıralaması |
| phase7_model_readiness_checklist | Model hazırlık kontrol listesi |
| phase7_feature_correlation_final | Final korelasyon özeti |

### CSV Raporlar (reports/csv/)

- `phase1_data_overview.csv`
- `phase2_univariate_stats.csv`
- `phase3_bivariate_stats.csv`
- `phase4_correlation_matrix.csv`
- `phase5_outlier_analysis.csv`
- `phase5_zscore_outliers.csv`
- `phase5_data_quality_summary.csv`
- `phase6_feature_importance_summary.csv`
- `phase6_all_data_prep_recommendations.csv`
- `phase7_model_readiness_checklist.csv`
- `phase7_data_quality_scorecard.csv`

---

*EDA Expert — CRISP-DM Data Understanding Aşaması tamamlandı.*
*Bir sonraki aşama: Data Preparation Expert*
