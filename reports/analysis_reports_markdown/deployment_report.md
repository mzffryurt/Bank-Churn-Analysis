# Deployment Expert Raporu
**Proje:** Musteri Churn Tahmin Sistemi  
**Model:** Logistic Regression (class_weight='balanced')  
**Olusturulma Tarihi:** 2026-05-30  
**Uygulama:** `app/app.py`

---

## 1. Yonetici Ozeti

Bu rapor, Model Expert tarafindan uretilen Logistic Regression modelinin Streamlit tabanli profesyonel bir uretim arayzuyle yayina alinma surecini belgelemektedir. Uygulama; tekil musteri tahmini, toplu CSV tahmini, model performans gosterimi, monitoring ve yardim dokumantasyonunu tek bir arayuzde birlestirir.

**Amaç:** Banka musterilerinin churn edip etmeyecegini (%20 azinlik sinifi) tahmin etmek; churn riski yuksek musterileri erken tespit ederek retention aksiyonu almak.

**Temel Karar:** Precision/Recall dengesi kasten Recall lehine kurgulanmistir. F1=0.4866, Recall=0.6716. Bu, "churn eden musteriyi kacirma" odakli bir is stratejisine uygundur.

---

## 2. Kullanilan Model ve Pipeline

| Bilesen         | Dosya                               | Aciklama                        |
|-----------------|-------------------------------------|---------------------------------|
| Final Model     | `models/final_model.pkl`            | LogisticRegression (balanced)   |
| Preprocessing   | `models/preprocessing_pipeline.pkl`| ColumnTransformer + OHE + Scaler|
| Model Sonuclari | `models/ranked_model_results.csv`   | 16 model karsilastirma          |

**Pipeline Beklentisi (12 sutun):**

| Sutun                 | Donusum       | Kaynak              |
|-----------------------|---------------|---------------------|
| CreditScore           | StandardScaler| Ham girdi           |
| Age                   | StandardScaler| Ham girdi           |
| Tenure                | StandardScaler| Ham girdi           |
| Balance               | StandardScaler| Ham girdi           |
| EstimatedSalary       | StandardScaler| Ham girdi           |
| Geography             | OneHotEncoder | Ham girdi           |
| Gender                | OneHotEncoder | Ham girdi           |
| NumOfProducts         | Passthrough   | Ham girdi           |
| HasCrCard             | Passthrough   | Ham girdi           |
| IsActiveMember        | Passthrough   | Ham girdi           |
| is_high_products_risk | Passthrough   | App icerisinde turetilmis (NumOfProducts>=3) |
| has_zero_balance      | Passthrough   | App icerisinde turetilmis (Balance==0)       |

Uygulama, kullanicidan yalnizca ham 10 sutunu alir; 2 turetilmis ozellik app.py icerisinde otomatik hesaplanir.

**Model Cikti Feature Isimleri (13 ozellik):**
`CreditScore, Age, Tenure, Balance, EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male, NumOfProducts, HasCrCard, IsActiveMember, is_high_products_risk, has_zero_balance`

---

## 3. Streamlit UI Mimarisi

### Sayfa Yapisi (6 Sekme)

| Sekme              | Icerik                                                               |
|--------------------|----------------------------------------------------------------------|
| Tekil Tahmin       | Form, ornek veri butonlari, reset, tahmin sonucu + gauge + bar chart |
| Toplu Tahmin (CSV) | Dosya yukleme, dogrulama, batch predict, ozet istatistikler, indirme |
| Model Performansi  | Ranked model tablosu, HTML/PNG performans grafikleri                 |
| Model Bilgisi      | Metrik kartlar, secim gerekceleri, pipeline tablosu, rapor goruntule |
| Monitoring         | Log tablosu, confidence histogram, model versiyon, drift onerisi     |
| Yardim             | Kullanim kilavuzu, alan aciklamalari, sinirliliklar, sorumluluk notu |

### Teknik Bilesenler
- `@st.cache_resource` ile model ve pipeline tek seferinde yuklenir
- `st.session_state` ile son tahmin, batch sonuc ve form sifirlama yonetilir
- Plotly ile interaktif gauge, bar ve histogram grafikleri
- `streamlit.components.v1.html` ile Model Expert'ten gelen HTML figurler gosterilir
- CSV loglama: `logs/prediction_log.csv` (timestamp, prediction, confidence, girdiler)

---

## 4. Shneiderman'in 8 Altin Kurali - Uygulama Kararlari

### Kural 1: Tutarlilik
Tum uygulama boyunca tek bir renk paleti kullanildi:
- Birincil mavi (#2E86AB), basari yesili (#6A994E), uyari turuncu (#F18F01), tehlike kirmizi (#C73E1D)
- Tum baslik hiyerarsisi (h1/h2/h3) ayni renk ve font agirligiyla tanimli
- Metric kartlar, hero kartlar ve sonuc kartlari tutarli CSS siniflarla (`hero-card`, `metric-card`, `result-positive/warning/danger`) stillendirildi
- Sidebar'da sabit model ozeti, risk seviyeleri ve versiyon bilgisi her sekmede gorunur kalir

### Kural 2: Evrensel Kullanilabilirlik / Kisayollar
- "Ornek: Dusuk Riskli Musteri" ve "Ornek: Yuksek Riskli Musteri" butonlari formu bir tikla doldurur
- "Ornek CSV Indir" butonu ile toplu tahmin formatini gostermek icin hazir sablonlu CSV indirilebilir
- Sidebar'daki risk seviyeleri efsanesi, kullanicinin tahmin sonucunu yorumlamak icin onemli bilgiyi akilda tutmaya gerek kalmaksizin gorebilmesini saglar

### Kural 3: Bilgilendirici Geri Bildirim
- Form gonderildikten sonra `st.spinner("Model hesapliyor...")` gosterilir
- Tahmin tamamlaninca sinif etiketi, churn yuzdesi ve risk seviyesi anlinda guncellenir
- Eylem onerisi kutucugu (`st.success/warning/error`) her tahmin icin ilgili aksiyonu aciklar
- Toplu tahminde progress bar satir bazinda ilerlemeyi gosterir
- CSV dogrulama hatalari, neyin yanlis oldugunu ve nasil duzeltilecegini acik sekilde belirtir

### Kural 4: Diyalogun Kapanisi (Closure)
Tekil tahmin akisi net bolumlere ayrildi:
1. Veri girisi (form)
2. Onizleme / dogrulama (turetilmis ozellik bilgilendirme kutulari)
3. Tahmin butonu
4. Sonuc gosterimi (sinif, olasilik, risk seviyesi)
5. Eylem onerisi (retention aksiyonu)
6. Gorsel pekistirme (gauge + bar chart)

Toplu tahminde ozet istatistik paneli (toplam / churn / retain / ort. olasilik) tamamlanmayi teyit eder.

### Kural 5: Hata Onleme ve Basit Hata Yonetimi
- Tum sayisal alanlarda `min_value` / `max_value` siniri tanimlanmis (orn. CreditScore 300-850, Age 18-95)
- Kategorik alanlarda sadece gecerli secenekler sunulur (selectbox)
- CSV yuklemede: eksik sutun tespiti, ek sutun uyarisi, veri tipi dogrulama, gecersiz kategori kontrolu, eksik deger kaldirilmasi
- Turetilmis ozellikler (is_high_products_risk, has_zero_balance) aktif oldugunda form icerisinde bilgi/uyari kutucugu gosterilir
- Model veya pipeline yuklenemezse anlasilir hata mesajiyla uygulama durdurulur (`st.stop()`)

### Kural 6: Geri Alinabilir Eylemler
- "Formu Sifirla" butonu tum form degerlerini ve son tahmini session state'ten temizler
- "Toplu Tahmin Sonuclarini Temizle" butonu batch sonucu kaldirir ve sayfayi yeniler
- Tahmin gecmisi session state'te tutuldugu icin yeni bir tahmin onceki sonucu ezmez; kullanici sonucu inceledikten sonra sifirlama kararini kendi verir

### Kural 7: Ic Kontrol Odagi
- Kullanici her zaman hangi risk bolgede oldugunu ve ne anlama geldigini sidebar'dan izler
- Gauge ve bar chart, sayisal degerleri gorsel olarak anlamlandirmaya yardimci olur
- Turetilmis ozellik gostergeleri, modelin arka planda ne hesapladigini transparan kilar
- Batch modda tum parametre kontrolleri kullanicinin elindedir; model sadece istege gore tetiklenir

### Kural 8: Kisa Sureli Bellek Yukunu Azaltma
- Sidebar'da her sekme degisiminde gorunen sabit model ozeti (F1, Recall, ROC-AUC)
- Risk seviyeleri renk-eslesme efsanesi sidebar'da kalici
- Yardim sekmesinde her alan icin aciklama ve aralik bilgisi
- Toplu tahmin sekmesinde beklenen sutun listesi ve ornek CSV indirme
- Tahmin sonrasinda eylem onerisi kutucugu kullanicinin "simdi ne yapmali" sorusuna aninda yanit verir

---

## 5. HCI Ilkelerine Gore Kullanilabilirlik Degerlendirmesi

### Nielsen Kullanilabilirlik Ilkeleri

| Ilke                              | Uygulama                                                        |
|-----------------------------------|-----------------------------------------------------------------|
| Sistem durumunun gorünürlügü      | Progress bar, spinner, log sayacı, model versiyon bilgisi       |
| Gercek dunyayla uyum              | "Churn Edecek/Etmeyecek" yerine "Musteri kaybedilebilir" dili   |
| Kullanici kontrolu ve ozgurlugu   | Reset butonu, batch temizle, sekme bazli navigasyon             |
| Tutarlilik ve standartlar         | Tek CSS sistemi, tutarli kart yapisi, renk kodlama              |
| Hata onleme                       | Input sinirlar, selectbox, CSV dogrulama                        |
| Hatirlama yerine tanima           | Ornek butonlar, sidebar efsane, tooltip/help text               |
| Esneklik ve verimlilik            | Ornek musteri butonlari, toplu CSV modu                         |
| Estetik ve minimalist tasarim     | Beyaz arka plan, genis bosluk, tek vurgu rengi                  |
| Hata tanima ve cozum              | Anlasilir hata mesajlari, ne yapilacagini gosteren uyarilar     |
| Yardim ve dokumantasyon           | Yardim sekmesi: kullanim kilavuzu + alan aciklamalari           |

### Don Norman - Execution/Evaluation Korfezi
- **Gulf of Execution azaltildi:** Her butonun amaci nettir; "Churn Tahmini Yap" tek ana aksiyon
- **Gulf of Evaluation azaltildi:** Gauge, renk kodlama ve eylem onerisi sonucun ne anlama geldigini aciklar

### Bilissel Yuk
- Teknik detaylar (pipeline ozellikleri, rapor metni) `st.expander` ile gizlenebilir durumda
- Sekme yapisi bilgiyi kategorize eder; kullanici tek ekranda asilmaz
- Toplu tahmin ozet kartlari (4 metrik), 5000 satirin anlamini tek bakista verir

---

## 6. Tahmin Akisi

### Tekil Tahmin
1. Kullanici 10 ham girdi girer (veya ornek butonla otomatik doldurur)
2. App `is_high_products_risk` ve `has_zero_balance` turetir
3. 12 sutunluk DataFrame pipeline'a verilir
4. Pipeline StandardScaler + OHE ile 13 ozellige donusturur
5. LogisticRegression `predict` + `predict_proba` calisir
6. Sinif (0/1), churn olasiligi, risk seviyesi ve eylem onerisi gosterilir
7. Log dosyasina yazilir

### Toplu Tahmin
1. CSV yuklenir, kolon/tip/kategori dogrulamasi yapilir
2. Chunk bazli (satir sayisi/20 batch) isleme
3. Her chunk icin turetilmis ozellikler hesaplanir, pipeline transform edilir, tahmin yapilir
4. Sonuclar orijinal DataFrame'e eklenir (Churn_Tahmini, Churn_Olasiligi, Risk_Seviyesi)
5. Ozet istatistik paneli, risk dagilimi grafigi ve indirilebilir CSV sunulur

---

## 7. Performans ve Model Bilgisi Gosterimi

Asagidaki figur dosyalari `tab_performance()` icerisinde gosterilmektedir:

| Dosya                                      | Sekme        |
|--------------------------------------------|--------------|
| model_phase10_final_confusion_matrix.html  | Model Perf.  |
| model_phase10_roc_all_models.html          | Model Perf.  |
| model_phase10_precision_recall_curve.html  | Model Perf.  |
| model_phase7_performance_comparison.html   | Model Perf.  |
| model_phase7_cv_stability.html             | Model Perf.  |
| model_phase7_overfitting_analysis.html     | Model Perf.  |
| model_phase7_leadership_matrix.html        | Model Perf.  |
| model_phase7_training_time.html            | Model Perf.  |

HTML yoksa ilgili PNG otomatik gosterilir. Model Bilgisi sekmesinde `model_expert_report.md` genisletilebilir panel icerisinde sunulur.

---

## 8. Monitoring ve Loglama

**Log Dosyasi:** `logs/prediction_log.csv`

**Loglanan Alanlar:**
- timestamp (ISO format)
- source ("single" / "batch")
- prediction (0/1)
- churn_prob (0.0-1.0)
- Tum ham girdi sutunlari

**Monitoring Sayfasi Gosterimleri:**
- Model versiyonu ve guncelleme tarihi
- Toplam loglanan tahmin sayisi
- Son 50 tahmin tablosu (en yeni ustte)
- Confidence dagilimi histogrami (log gecmisi > 5 ise)
- Log dosyasi indirme
- Model drift izleme onerileri

---

## 9. Guvenlik, Etik ve Sinirliliklar

- Model karari tek basina nihai is karari degildir; yuksek riskli durumlarda uzman degerlendirmesi onerilir
- Dusuk guvenli tahminler (olasilik %40-60 arasi) "Orta Risk" olarak isaretlenmistir
- Gender ve Geography gibi hassas degiskenler modelde kullanilmakta; ayrimcilik riski degerlendirilmelidir
- Kullanici verileri loglaniyorsa KVKK/GDPR uyumlulugu saglanmalidir
- Precision dusuk (%38): "churn edecek" tahminlerinin %62'si yanlis pozitif olabilir; iletisim maliyeti degerlendirilmelidir
- Model yili itibarayla 6-12 ayda bir yeniden egitim gerekebilir (musteri davranis drift'i)

---

## 10. Sonraki Adimlar

### Kisa Vade
1. `streamlit run app/app.py` ile uygulama baslatilir
2. `logs/prediction_log.csv` duzenli izlenir
3. Confidence dagiliminda kayma gorulurse model drift analizi yapilir

### Orta Vade
4. Gercek churn etiketi toplanan verilere gore dogruluk kontrolu
5. Threshold ayari (varsayilan 0.5 yerine 0.35-0.40 degerlendir)
6. A/B testi: hangi risk segmentinde retention kampanyasi daha etkili?

### Uzun Vade
7. Model yeniden egitimi (yeni musteri verisiyle)
8. Monitoring Expert agenti entegrasyonu (otomatik drift alarm)
9. API endpoint olarak servis haline getirme (FastAPI + Docker)

---

## Calistirma Talimatlari

```bash
# Proje dizininde:
pip install -r requirements.txt
streamlit run app/app.py
```

Tarayici otomatik acar; varsayilan adres: http://localhost:8501

**Dizin Yapisi:**
```
churn-analysis/
  app/
    app.py
  models/
    final_model.pkl
    preprocessing_pipeline.pkl
    ranked_model_results.csv
  figures/
    model_phase10_*.html
    model_phase7_*.html
  reports/markdown/
    model_expert_report.md
    deployment_report.md
  logs/
    prediction_log.csv   (otomatik olusturulur)
  requirements.txt
```
