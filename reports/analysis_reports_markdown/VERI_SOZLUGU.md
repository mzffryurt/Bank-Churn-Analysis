# 📊 Bank Customer Churn — Veri Sözlüğü

**Veri seti:** `data/raw/churn.csv`
**Gözlem sayısı:** 10.000 satır
**Değişken sayısı:** 14 sütun
**Hedef değişken:** `Exited` (1 = müşteri bankadan ayrıldı, 0 = kaldı)
**Problem tipi:** Binary Classification
**Churn oranı:** ~%20 (hafif sınıf dengesizliği — "hata maliyeti" tartışması için ideal)

---

## İş Problemi (Business Understanding için)

Banka, müşterilerini kaybetmeden (churn) **önce** kimin ayrılma riski taşıdığını bilmek istiyor.
Doğru tahmin → riskli müşteriye zamanında kampanya/teklif → müşteri elde tutma (retention).
Yeni müşteri kazanmak, mevcut müşteriyi tutmaktan çok daha pahalıdır; bu yüzden churn tahmini doğrudan kâra etki eder.

**Karar sorusu:** Hangi müşteri ayrılacak ve banka kimi öncelikli olarak elde tutmaya çalışmalı?

---

## Sütun Açıklamaları

| Sütun | Tip | Açıklama | Modelleme Notu |
|---|---|---|---|
| `RowNumber` | int | Sıra numarası | **Modelden çıkar** — anlamsız index |
| `CustomerId` | int | Müşteri kimlik numarası | **Modelden çıkar** — kimlik, tahmin gücü yok |
| `Surname` | metin | Müşteri soyadı | **Modelden çıkar** — leakage/anlamsız |
| `CreditScore` | int | Kredi skoru (350–850) | Sayısal — scaling gerekebilir |
| `Geography` | kategorik | Ülke: France / Germany / Spain | One-Hot Encoding |
| `Gender` | kategorik | Cinsiyet: Male / Female | One-Hot / Label Encoding |
| `Age` | int | Yaş (18–92) | Churn'ün **en güçlü** belirleyicilerinden |
| `Tenure` | int | Bankada kalma süresi (0–10 yıl) | Sayısal |
| `Balance` | float | Hesap bakiyesi | ~%36'sı sıfır — dağılım çarpık olabilir |
| `NumOfProducts` | int | Kullanılan banka ürünü sayısı (1–4) | 3-4 ürünlülerde churn çok yüksek |
| `HasCrCard` | ikili | Kredi kartı var mı (1/0) | Zaten 0/1 |
| `IsActiveMember` | ikili | Aktif üye mi (1/0) | Aktif olmayanlar daha çok churn eder |
| `EstimatedSalary` | float | Tahmini yıllık maaş | Sayısal — scaling gerekebilir |
| `Exited` | ikili | **HEDEF** — ayrıldı mı (1/0) | Tahmin edilecek değişken |

---

## Veride Gözlenen Gerçekçi Paternler (EDA'da bunları bulacaksınız)

- **Almanya** müşterileri (%~29), Fransa ve İspanya'ya (%~17) göre belirgin biçimde daha çok ayrılıyor.
- **Aktif olmayan üyeler** (%~28), aktif üyelere (%~13) göre iki kat fazla churn ediyor.
- **3 veya 4 ürünü** olan müşterilerde churn oranı çok yüksek (%~60+).
- **Yaş** arttıkça ayrılma olasılığı artıyor.

> Bu paternler iş yorumunuzun (Evaluation fazı) iskeletini oluşturur:
> "Model neden bu tahmini yapıyor ve banka hangi aksiyonu almalı?"

---

## ⚠️ Önemli Not: Gerçek Veri Seti

Bu dosya, dünyaca bilinen **"Churn_Modelling" (Bank Customer Churn)** veri setiyle **birebir aynı şemada ve gerçekçi dağılımlarda** hazırlanmış 10.000 satırlık bir versiyondur; hocanın ajan kodları üzerinde sorunsuz çalışır.

Eğer **birebir orijinal (gerçek) dosyayı** kullanmak isterseniz, aynı sütunlarla şuradan indirip `data/raw/churn.csv` ile değiştirmeniz yeterli (kod değişmez):

- Kaggle: "Churn Modelling" / "Bank Customer Churn" (10.000 satır)
- GitHub (raw): bkz. README'deki kaynak linkleri

Hocaya **10.000 satır şartının bu churn seti için geçerli olup olmadığını** mutlaka teyit ettirin.
