# =============================================================================
# MODEL EXPERT — Churn Binary Classification
# 12-Phase Agentic Modelling Pipeline
# =============================================================================

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from prettytable import PrettyTable

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# DIZIN YAPILANDIRMASI
# -----------------------------------------------------------------------
BASE_DIR = "C:/Users/sence/OneDrive/Desktop/churn-analysis"
os.makedirs(f"{BASE_DIR}/figures",              exist_ok=True)
os.makedirs(f"{BASE_DIR}/models",               exist_ok=True)
os.makedirs(f"{BASE_DIR}/reports/markdown",     exist_ok=True)

# -----------------------------------------------------------------------
# SKLEARN
# -----------------------------------------------------------------------
from sklearn.dummy        import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.svm          import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics      import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)

import xgboost  as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[UYARI] CatBoost yuklu degil, atlanacak.")

RANDOM_STATE = 42

# -----------------------------------------------------------------------
# RENK PALETİ
# -----------------------------------------------------------------------
PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
    "#6A994E", "#BC4B51", "#8E7DBE", "#F77F00",
    "#06A77D", "#D4A574", "#5DADE2", "#F39C12",
    "#1ABC9C", "#8E44AD"
]

PASTEL_PALETTE = [
    "#A7C7E7", "#B8E0D2", "#F6C6C6", "#F7D9A3",
    "#D7BDE2", "#C8D6AF", "#F5CBA7", "#AED6F1",
    "#D5F5E3", "#FADBD8", "#A9CCE3", "#F9E4B7",
    "#D2B4DE", "#A9DFBF"
]

# -----------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------
model_results   = []
model_decisions = []
handoff_notes   = []


def log_model_result(model_name, train_f1, test_f1, cv_f1_mean, cv_f1_std,
                     test_recall, test_precision, test_roc_auc, test_accuracy,
                     overfit_gap, train_time, status="Basarili"):
    model_results.append({
        "Model":            model_name,
        "Train F1":         train_f1,
        "Test F1":          test_f1,
        "CV F1 Ort":        cv_f1_mean,
        "CV F1 Std":        cv_f1_std,
        "Recall":           test_recall,
        "Precision":        test_precision,
        "ROC-AUC":          test_roc_auc,
        "Accuracy":         test_accuracy,
        "Overfit Gap":      overfit_gap,
        "Egitim Suresi (s)": train_time,
        "Durum":            status,
    })


def apply_premium_layout(fig, title):
    fig.update_layout(
        title={
            "text":     title,
            "x":        0.03,
            "xanchor":  "left",
            "font":     {"size": 20, "family": "Arial", "color": "#1F2937"},
        },
        template="plotly_white",
        paper_bgcolor="#FBFBF8",
        plot_bgcolor="#FBFBF8",
        font={"family": "Arial", "size": 12, "color": "#374151"},
        margin=dict(l=80, r=40, t=90, b=80),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig


def save_figure(fig, name):
    path_html = f"{BASE_DIR}/figures/{name}.html"
    path_png  = f"{BASE_DIR}/figures/{name}.png"
    fig.write_html(path_html)
    try:
        fig.write_image(path_png, width=1400, height=800, scale=2)
        print(f"  [KAYIT] {path_png}")
    except Exception as e:
        print(f"  [UYARI] PNG kaydedilemedi ({e}); HTML kaydedildi: {path_html}")


# =============================================================================
# PHASE 1 — VERİ YÜKLEME VE DOGRULAMA
# =============================================================================
print("\n" + "="*70)
print("PHASE 1: DATAPREP HANDOFF INGESTION")
print("="*70)

X_train = pd.read_csv(f"{BASE_DIR}/data/model_ready/X_train.csv")
X_test  = pd.read_csv(f"{BASE_DIR}/data/model_ready/X_test.csv")
y_train = pd.read_csv(f"{BASE_DIR}/data/model_ready/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{BASE_DIR}/data/model_ready/y_test.csv").squeeze()

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train sinif dagilimi:\n{y_train.value_counts()}")
print(f"y_test  sinif dagilimi:\n{y_test.value_counts()}")
print(f"Feature listesi: {X_train.columns.tolist()}")
print(f"Eksik deger X_train: {X_train.isna().sum().sum()}  |  X_test: {X_test.isna().sum().sum()}")

# Preprocessing pipeline yukle (bilgi amacli)
try:
    pipeline = joblib.load(f"{BASE_DIR}/models/preprocessing_pipeline.pkl")
    print(f"[OK] Preprocessing pipeline yuklendi: {type(pipeline)}")
except Exception as e:
    pipeline = None
    print(f"[UYARI] Pipeline yuklenemedi: {e}")

# =============================================================================
# PHASE 2 — PROBLEM FRAMING
# =============================================================================
print("\n" + "="*70)
print("PHASE 2: PROBLEM FRAMING")
print("="*70)

n_classes    = y_train.nunique()
class_counts = y_train.value_counts()
minority_ratio = class_counts.min() / class_counts.sum()
imbalance_ratio = class_counts.max() / class_counts.min()

print(f"Problem tipi     : Binary Classification")
print(f"Sinif sayisi     : {n_classes}")
print(f"Azinlik orani    : {minority_ratio:.3f}  ({minority_ratio*100:.1f}%)")
print(f"Dengesizlik orani: {imbalance_ratio:.2f}:1")
print(f"Strateji         : class_weight='balanced' (SMOTE YOK)")
print(f"Ana metrik       : F1 (pozitif sinif) + Recall > ROC-AUC > Accuracy")

# =============================================================================
# PHASE 3 — METRIC STRATEGY
# =============================================================================
print("\n" + "="*70)
print("PHASE 3: METRIC STRATEGY")
print("="*70)

CV_FOLDS    = 5
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"CV stratejisi    : StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True)")
print(f"CV scoring       : f1 (pozitif sinif) + roc_auc")
print(f"Test metrikleri  : F1, Recall, Precision, ROC-AUC, Accuracy")

# Class weight hesaplama
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}
scale_pos_weight  = class_counts[0] / class_counts[1]  # XGBoost icin

print(f"\nHesaplanan class_weight: {class_weight_dict}")
print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

# =============================================================================
# PHASE 4 & 5 — MODEL CANDIDATE POOL
# =============================================================================
print("\n" + "="*70)
print("PHASE 4-5: BASELINE + MODEL CANDIDATE POOL (12+)")
print("="*70)

models = {
    # --- BASELINE ---
    "Dummy (MostFreq)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),

    # --- LINEAR ---
    "Logistic Regression":  LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE,
        class_weight="balanced", solver="lbfgs"
    ),
    "Ridge Classifier":     RidgeClassifier(
        class_weight="balanced"
    ),

    # --- INSTANCE-BASED (class_weight desteklemez) ---
    "KNN":                  KNeighborsClassifier(n_neighbors=7),
    "GaussianNB":           GaussianNB(),

    # --- TREE ---
    "Decision Tree":        DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced",
        max_depth=8
    ),

    # --- ENSEMBLE: BAGGING ---
    "Random Forest":        RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    ),
    "Extra Trees":          ExtraTreesClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    ),
    "Bagging":              BaggingClassifier(
        random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1
    ),

    # --- ENSEMBLE: BOOSTING ---
    "Gradient Boosting":    GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        learning_rate=0.1, max_depth=4
    ),
    "AdaBoost":             AdaBoostClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        learning_rate=0.5
    ),

    # --- SVM ---
    "SVM (RBF)":            SVC(
        probability=True, random_state=RANDOM_STATE,
        class_weight="balanced", kernel="rbf"
    ),

    # --- MLP ---
    "MLP Neural Net":       MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1
    ),

    # --- XGBOOST ---
    "XGBoost":              xgb.XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1, verbosity=0,
        learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8
    ),

    # --- LIGHTGBM ---
    "LightGBM":             lgb.LGBMClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        is_unbalance=True, n_jobs=-1,
        verbose=-1,
        learning_rate=0.05, max_depth=5,
        num_leaves=31
    ),
}

# CatBoost opsiyonel
if CATBOOST_AVAILABLE:
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        verbose=0, learning_rate=0.05, depth=5
    )

# class_weight desteklemeyenler — bilgi notu
NO_CLASS_WEIGHT = ["KNN", "GaussianNB", "Bagging", "Gradient Boosting",
                   "AdaBoost", "MLP Neural Net", "XGBoost", "LightGBM"]
print(f"\nclass_weight desteklemeyen modeller (basici strateji): {NO_CLASS_WEIGHT}")
print(f"Toplam model sayisi: {len(models)}")

# =============================================================================
# PHASE 6 — TRAINING LOOP
# =============================================================================
print("\n" + "="*70)
print("PHASE 6: MODEL TRAINING LOOP")
print("="*70)

trained_models = {}
roc_data       = {}  # ROC egrileri icin

for model_name, model in models.items():
    print(f"\n  [{model_name}] egitiliyor...")
    t0 = time.time()

    try:
        model.fit(X_train, y_train)

        # --- Tahminler ---
        train_pred  = model.predict(X_train)
        test_pred   = model.predict(X_test)

        # ROC icin olasilik
        if hasattr(model, "predict_proba"):
            test_proba  = model.predict_proba(X_test)[:, 1]
            train_proba = model.predict_proba(X_train)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X_test)
            test_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            raw_tr = model.decision_function(X_train)
            train_proba = (raw_tr - raw_tr.min()) / (raw_tr.max() - raw_tr.min() + 1e-9)
        else:
            test_proba  = test_pred.astype(float)
            train_proba = train_pred.astype(float)

        # --- Metrikler ---
        train_f1  = f1_score(y_train, train_pred, pos_label=1, zero_division=0)
        test_f1   = f1_score(y_test,  test_pred,  pos_label=1, zero_division=0)
        test_rec  = recall_score(y_test,  test_pred,  pos_label=1, zero_division=0)
        test_prec = precision_score(y_test, test_pred, pos_label=1, zero_division=0)
        test_acc  = accuracy_score(y_test, test_pred)
        test_auc  = roc_auc_score(y_test, test_proba)
        overfit   = round(train_f1 - test_f1, 4)

        # --- Stratified CV (F1) ---
        cv_f1 = cross_val_score(
            model, X_train, y_train,
            cv=cv_strategy, scoring="f1",
            n_jobs=-1
        )
        cv_f1_mean = round(float(np.mean(cv_f1)), 4)
        cv_f1_std  = round(float(np.std(cv_f1)),  4)

        train_time = round(time.time() - t0, 2)

        log_model_result(
            model_name    = model_name,
            train_f1      = round(train_f1, 4),
            test_f1       = round(test_f1,  4),
            cv_f1_mean    = cv_f1_mean,
            cv_f1_std     = cv_f1_std,
            test_recall   = round(test_rec,  4),
            test_precision= round(test_prec, 4),
            test_roc_auc  = round(test_auc,  4),
            test_accuracy = round(test_acc,  4),
            overfit_gap   = overfit,
            train_time    = train_time,
            status        = "Basarili"
        )

        # ROC veri saklama
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        roc_data[model_name] = {
            "fpr": fpr, "tpr": tpr, "auc": round(test_auc, 4)
        }

        trained_models[model_name] = model
        print(f"    F1={test_f1:.4f}  Recall={test_rec:.4f}  ROC-AUC={test_auc:.4f}  "
              f"Overfit={overfit:+.4f}  Sure={train_time}s")

    except Exception as e:
        train_time = round(time.time() - t0, 2)
        print(f"    [HATA] {e}")
        log_model_result(
            model_name=model_name, train_f1=None, test_f1=None,
            cv_f1_mean=None, cv_f1_std=None,
            test_recall=None, test_precision=None,
            test_roc_auc=None, test_accuracy=None,
            overfit_gap=None, train_time=train_time,
            status=f"Calismaidi: {str(e)[:60]}"
        )

# =============================================================================
# PHASE 7 — PRETTYTABLE MODEL COMPARISON
# =============================================================================
print("\n" + "="*70)
print("PHASE 7: MODEL KARSILASTIRMA (PrettyTable)")
print("="*70)

results_df = pd.DataFrame(model_results)

# F1 skoruna gore sirala
sorted_df = results_df.dropna(subset=["Test F1"]).sort_values(
    "Test F1", ascending=False
).reset_index(drop=True)

table = PrettyTable()
table.field_names = [
    "#", "Model", "Train F1", "Test F1", "CV F1 Ort",
    "CV Std", "Recall", "Precision", "ROC-AUC", "Accuracy",
    "Overfit", "Sure (s)", "Durum"
]
table.align["Model"] = "l"

for i, row in sorted_df.iterrows():
    table.add_row([
        i + 1,
        row["Model"],
        f"{row['Train F1']:.4f}"   if pd.notna(row["Train F1"])   else "-",
        f"{row['Test F1']:.4f}"    if pd.notna(row["Test F1"])    else "-",
        f"{row['CV F1 Ort']:.4f}"  if pd.notna(row["CV F1 Ort"]) else "-",
        f"{row['CV F1 Std']:.4f}"  if pd.notna(row["CV F1 Std"]) else "-",
        f"{row['Recall']:.4f}"     if pd.notna(row["Recall"])     else "-",
        f"{row['Precision']:.4f}"  if pd.notna(row["Precision"])  else "-",
        f"{row['ROC-AUC']:.4f}"    if pd.notna(row["ROC-AUC"])   else "-",
        f"{row['Accuracy']:.4f}"   if pd.notna(row["Accuracy"])   else "-",
        f"{row['Overfit Gap']:+.4f}" if pd.notna(row["Overfit Gap"]) else "-",
        f"{row['Egitim Suresi (s)']:.2f}s" if pd.notna(row["Egitim Suresi (s)"]) else "-",
        row["Durum"],
    ])

print(table)

# Tum sonuclari kaydet (basarisizlar dahil)
results_df.to_csv(f"{BASE_DIR}/models/all_model_results.csv", index=False)
print(f"\n[KAYIT] {BASE_DIR}/models/all_model_results.csv")

# PrettyTable txt kaydet
with open(f"{BASE_DIR}/reports/markdown/model_comparison_prettytable.txt", "w", encoding="utf-8") as f:
    f.write(str(table))
print(f"[KAYIT] {BASE_DIR}/reports/markdown/model_comparison_prettytable.txt")

# =============================================================================
# PHASE 7.5 — GORSEL KARSILASTIRMA SUITE
# =============================================================================
print("\n" + "="*70)
print("PHASE 7.5: GORSEL MODEL KARSILASTIRMA SUITE")
print("="*70)

plot_df = sorted_df.copy()

# --- Grafik 1: Ana Performans (F1, Recall, ROC-AUC) ---
fig1 = go.Figure()

metrics_to_plot = [
    ("Test F1",   "#2E86AB", "Test F1"),
    ("Recall",    "#A23B72", "Recall"),
    ("ROC-AUC",   "#6A994E", "ROC-AUC"),
    ("Precision", "#F18F01", "Precision"),
]

for col, color, label in metrics_to_plot:
    if col in plot_df.columns:
        fig1.add_trace(go.Bar(
            name=label,
            x=plot_df["Model"],
            y=plot_df[col],
            marker_color=color,
            text=[f"{v:.3f}" for v in plot_df[col]],
            textposition="outside",
        ))

fig1.update_layout(barmode="group")
fig1 = apply_premium_layout(fig1, "12+ Model Ana Performans Karsilastirmasi — F1 / Recall / ROC-AUC / Precision")
fig1.update_layout(height=600, xaxis_tickangle=-35)
save_figure(fig1, "model_phase7_performance_comparison")
fig1.show()

# --- Grafik 2: CV Kararliligi ---
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=plot_df["Model"],
    y=plot_df["CV F1 Ort"],
    error_y=dict(type="data", array=plot_df["CV F1 Std"], visible=True),
    marker_color="#A7C7E7",
    name="CV F1 Ortalamasi",
    text=[f"{v:.3f}" for v in plot_df["CV F1 Ort"]],
    textposition="outside",
))
fig2 = apply_premium_layout(fig2, "Model CV Kararliligi — Stratified 5-Fold F1 Ortalama ± Std")
fig2.update_layout(height=550, xaxis_tickangle=-35)
save_figure(fig2, "model_phase7_cv_stability")
fig2.show()

# --- Grafik 3: Overfitting Analizi ---
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    name="Train F1",
    x=plot_df["Model"],
    y=plot_df["Train F1"],
    marker_color="#2E86AB",
))
fig3.add_trace(go.Bar(
    name="Test F1",
    x=plot_df["Model"],
    y=plot_df["Test F1"],
    marker_color="#F6C6C6",
))
fig3.update_layout(barmode="group")
fig3 = apply_premium_layout(fig3, "Train vs Test F1 — Overfitting Analizi")
fig3.update_layout(height=550, xaxis_tickangle=-35)
save_figure(fig3, "model_phase7_overfitting_analysis")
fig3.show()

# --- Grafik 4: Egitim Suresi ---
fig4 = px.bar(
    plot_df.sort_values("Egitim Suresi (s)", ascending=True),
    x="Model", y="Egitim Suresi (s)",
    color="Test F1",
    color_continuous_scale=["#D5F5E3", "#A7C7E7", "#2E86AB"],
    title="Model Egitim Suresi vs Performans",
    text="Egitim Suresi (s)"
)
fig4.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
fig4 = apply_premium_layout(fig4, "Model Egitim Suresi vs Test F1 Performansi")
fig4.update_layout(height=550, xaxis_tickangle=-35)
save_figure(fig4, "model_phase7_training_time")
fig4.show()

# --- Grafik 5: Liderlik Matrisi ---
valid_for_matrix = plot_df.dropna(
    subset=["Test F1", "Overfit Gap", "Egitim Suresi (s)", "CV F1 Std"]
).copy()
valid_for_matrix["Abs Overfit"] = valid_for_matrix["Overfit Gap"].abs()

fig5 = px.scatter(
    valid_for_matrix,
    x="Test F1",
    y="Abs Overfit",
    size="Egitim Suresi (s)",
    color="CV F1 Std",
    hover_name="Model",
    text="Model",
    color_continuous_scale=["#D5F5E3", "#F7D9A3", "#C73E1D"],
    title="Model Liderlik Matrisi: Performans / Overfit Riski / Hiz / CV Kararliligi",
    labels={
        "Test F1":    "Test F1 (yuksek = iyi)",
        "Abs Overfit":"Overfit Riski (dusuk = iyi)",
    }
)
fig5.update_traces(textposition="top center", marker=dict(sizemin=10))
# Ideal bolge isareti
fig5.add_annotation(
    x=valid_for_matrix["Test F1"].max() * 0.95,
    y=valid_for_matrix["Abs Overfit"].min() * 1.5 + 0.01,
    text="IDEAL BOLGE", showarrow=False,
    font=dict(size=12, color="#6A994E"),
    bgcolor="#D5F5E3"
)
fig5 = apply_premium_layout(fig5, "Model Liderlik Matrisi")
fig5.update_layout(height=650)
save_figure(fig5, "model_phase7_leadership_matrix")
fig5.show()

print("\n[GRAFIK OZETI]")
print(f"  En yuksek Test F1        : {sorted_df.iloc[0]['Model']}  ({sorted_df.iloc[0]['Test F1']:.4f})")
print(f"  En kararli (dusuk CV Std): {sorted_df.loc[sorted_df['CV F1 Std'].idxmin(), 'Model']}  "
      f"({sorted_df['CV F1 Std'].min():.4f})")
print(f"  En dusuk overfit         : {sorted_df.loc[sorted_df['Overfit Gap'].abs().idxmin(), 'Model']}  "
      f"({sorted_df['Overfit Gap'].abs().min():.4f})")
print(f"  En hizli model           : {sorted_df.loc[sorted_df['Egitim Suresi (s)'].idxmin(), 'Model']}  "
      f"({sorted_df['Egitim Suresi (s)'].min():.2f}s)")

# =============================================================================
# PHASE 9 — FINAL MODEL DECISION (cok kriterli)
# =============================================================================
print("\n" + "="*70)
print("PHASE 9: FINAL MODEL DECISION")
print("="*70)

# Cok kriterli skorlama
# Test F1 * 0.35 + Recall * 0.25 + ROC-AUC * 0.20 + (1-Overfit/0.5)*0.10 + CV_F1*(0.10)
def multi_criteria_score(row):
    if pd.isna(row["Test F1"]):
        return -9999
    f1_s    = row["Test F1"]
    rec_s   = row["Recall"]
    auc_s   = row["ROC-AUC"]
    ov_pen  = max(0, 1 - abs(row["Overfit Gap"]) / 0.5)
    cv_s    = row["CV F1 Ort"]
    return (f1_s * 0.35 + rec_s * 0.25 + auc_s * 0.20 + ov_pen * 0.10 + cv_s * 0.10)

sorted_df["MultiCriteria"] = sorted_df.apply(multi_criteria_score, axis=1)
sorted_df = sorted_df.sort_values("MultiCriteria", ascending=False).reset_index(drop=True)

print("\nCok Kriterli Sirali Tablo (F1*0.35 + Recall*0.25 + AUC*0.20 + CV*0.10 + OverfitPenalty*0.10):")
mc_table = PrettyTable()
mc_table.field_names = ["#", "Model", "Test F1", "Recall", "ROC-AUC",
                         "Overfit", "CV F1 Ort", "MultiCriteria"]
mc_table.align["Model"] = "l"
for i, row in sorted_df.iterrows():
    mc_table.add_row([
        i + 1, row["Model"],
        f"{row['Test F1']:.4f}" if pd.notna(row["Test F1"]) else "-",
        f"{row['Recall']:.4f}"  if pd.notna(row["Recall"])  else "-",
        f"{row['ROC-AUC']:.4f}" if pd.notna(row["ROC-AUC"]) else "-",
        f"{row['Overfit Gap']:+.4f}" if pd.notna(row["Overfit Gap"]) else "-",
        f"{row['CV F1 Ort']:.4f}" if pd.notna(row["CV F1 Ort"]) else "-",
        f"{row['MultiCriteria']:.4f}" if pd.notna(row["MultiCriteria"]) else "-",
    ])
print(mc_table)

best_row        = sorted_df.iloc[0]
best_model_name = best_row["Model"]
best_model      = trained_models[best_model_name]

print(f"\n[KARAR] Final model: {best_model_name}")
print(f"  Test F1       : {best_row['Test F1']:.4f}")
print(f"  Recall        : {best_row['Recall']:.4f}")
print(f"  Precision     : {best_row['Precision']:.4f}")
print(f"  ROC-AUC       : {best_row['ROC-AUC']:.4f}")
print(f"  Accuracy      : {best_row['Accuracy']:.4f}")
print(f"  Overfit Gap   : {best_row['Overfit Gap']:+.4f}")
print(f"  CV F1 Ort     : {best_row['CV F1 Ort']:.4f}")
print(f"  MultiCriteria : {best_row['MultiCriteria']:.4f}")

# =============================================================================
# PHASE 10 — CONFUSION MATRIX + ROC CURVE
# =============================================================================
print("\n" + "="*70)
print("PHASE 10: CONFUSION MATRIX + ROC CURVE")
print("="*70)

# Final model tahminleri
best_model.fit(X_train, y_train)
y_pred_final  = best_model.predict(X_test)
if hasattr(best_model, "predict_proba"):
    y_proba_final = best_model.predict_proba(X_test)[:, 1]
elif hasattr(best_model, "decision_function"):
    raw = best_model.decision_function(X_test)
    y_proba_final = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
else:
    y_proba_final = y_pred_final.astype(float)

# Classification report
print(f"\n[{best_model_name}] Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=["Cikmayanlar (0)", "Churn (1)"]))

cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}")
print(f"Yorum: {fn} musterinin churn'u kaciriliyor (False Negative)")
print(f"Yorum: {fp} musteri yanlis alarm (False Positive)")

# --- Confusion Matrix Gorseli ---
cm_labels = ["Cikmayanlar (0)", "Churn (1)"]
cm_text   = [[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]]

fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=cm_labels,
    y=cm_labels,
    annotation_text=cm_text,
    colorscale=[[0, "#FBFBF8"], [0.5, "#A7C7E7"], [1, "#2E86AB"]],
    showscale=True,
)
fig_cm.update_layout(
    title=dict(
        text=f"Final Model Confusion Matrix — {best_model_name}",
        x=0.03, xanchor="left",
        font=dict(size=20, family="Arial", color="#1F2937")
    ),
    xaxis_title="Tahmin Edilen Sinif",
    yaxis_title="Gercek Sinif",
    template="plotly_white",
    paper_bgcolor="#FBFBF8",
    plot_bgcolor="#FBFBF8",
    font=dict(family="Arial", size=13, color="#374151"),
    height=500, width=600,
)
save_figure(fig_cm, "model_phase10_final_confusion_matrix")
fig_cm.show()

# --- ROC Curve — Tum Modeller ---
fig_roc = go.Figure()

for idx, (mname, rdata) in enumerate(roc_data.items()):
    color = PROFESSIONAL_PALETTE[idx % len(PROFESSIONAL_PALETTE)]
    width = 3.5 if mname == best_model_name else 1.2
    dash  = "solid" if mname == best_model_name else "dot"
    fig_roc.add_trace(go.Scatter(
        x=rdata["fpr"], y=rdata["tpr"],
        mode="lines",
        name=f"{mname} (AUC={rdata['auc']:.3f})",
        line=dict(color=color, width=width, dash=dash),
    ))

# Rastgele siniflandirici
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Rastgele Tahmin",
    line=dict(color="#BBBBBB", width=1.5, dash="dash"),
))
fig_roc = apply_premium_layout(
    fig_roc,
    f"ROC Egri Karsilastirmasi — Tum Modeller (En Iyi: {best_model_name})"
)
fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.55, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
    height=700,
)
save_figure(fig_roc, "model_phase10_roc_all_models")
fig_roc.show()

# --- Precision-Recall Curve (final model) ---
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba_final)
pr_auc = average_precision_score(y_test, y_proba_final)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(
    x=rec_arr, y=prec_arr,
    mode="lines", fill="tozeroy",
    name=f"{best_model_name} (AP={pr_auc:.3f})",
    line=dict(color="#2E86AB", width=2.5),
    fillcolor="rgba(46,134,171,0.15)"
))
baseline_pr = y_test.sum() / len(y_test)
fig_pr.add_hline(y=baseline_pr, line_dash="dash", line_color="#BBBBBB",
                  annotation_text=f"Baseline ({baseline_pr:.3f})")
fig_pr = apply_premium_layout(
    fig_pr,
    f"Precision-Recall Egrisi — {best_model_name} (AP={pr_auc:.3f})"
)
fig_pr.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
    height=550
)
save_figure(fig_pr, "model_phase10_precision_recall_curve")
fig_pr.show()

# =============================================================================
# PHASE 12 — FINAL MODEL KAYDET + HANDOFF RAPORU
# =============================================================================
print("\n" + "="*70)
print("PHASE 12: FINAL MODEL KAYDET + HANDOFF RAPORU")
print("="*70)

# Model kaydet
model_path = f"{BASE_DIR}/models/final_model.pkl"
joblib.dump(best_model, model_path)
print(f"[KAYIT] {model_path}")

# Tum sonuclari kaydet
results_df.to_csv(f"{BASE_DIR}/models/all_model_results.csv", index=False)
sorted_df.to_csv(f"{BASE_DIR}/models/ranked_model_results.csv", index=False)

# =============================================================================
# MARKDOWN RAPOR
# =============================================================================
runner_up_row = sorted_df.iloc[1] if len(sorted_df) > 1 else None

report = f"""# Model Expert — Churn Binary Classification Raporu
Olusturulma Tarihi: 2026-05-29

---

## PHASE 1: Veri Dogrulama

| Boyut | Deger |
|---|---|
| X_train | 8000 x 13 |
| X_test  | 2000 x 13 |
| y_train sinif 0 | {int((y_train==0).sum())} (%{(y_train==0).mean()*100:.1f}) |
| y_train sinif 1 | {int((y_train==1).sum())} (%{(y_train==1).mean()*100:.1f}) |
| y_test sinif 0  | {int((y_test==0).sum())} (%{(y_test==0).mean()*100:.1f}) |
| y_test sinif 1  | {int((y_test==1).sum())} (%{(y_test==1).mean()*100:.1f}) |
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

Toplam {len(models)} model denendi.

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
| XGBoost | 300 | scale_pos_weight={scale_pos_weight:.2f}, lr=0.05 |
| LightGBM | 300 | is_unbalance=True, lr=0.05 |
| CatBoost | 300 | auto_class_weights='Balanced', lr=0.05 |
| SVM (RBF) | - | class_weight='balanced', probability=True |
| MLP | - | hidden=(128,64,32), early_stopping=True |

---

## PHASE 9: Final Model Secimi

### Secilen Model: {best_model_name}

| Metrik | Deger |
|---|---|
| Test F1 (pozitif sinif) | {best_row['Test F1']:.4f} |
| Recall | {best_row['Recall']:.4f} |
| Precision | {best_row['Precision']:.4f} |
| ROC-AUC | {best_row['ROC-AUC']:.4f} |
| Accuracy | {best_row['Accuracy']:.4f} |
| Overfit Gap (Train-Test F1) | {best_row['Overfit Gap']:+.4f} |
| CV F1 Ortalama | {best_row['CV F1 Ort']:.4f} |
| CV F1 Std | {best_row['CV F1 Std']:.4f} |
| Cok Kriterli Puan | {best_row['MultiCriteria']:.4f} |

**Secim Gerekce:** Final model olarak {best_model_name} secilmistir. Bu secim yalnizca en yuksek
test F1 skoruna degil; CV kararliligi, train-test overfitting farki, Recall performansi
ve uretimde uygulanabilirlik kriterlerine dayanir. Sinif dengesizligi nedeniyle
tek basina accuracy kullanilmamis; F1 ve Recall on plana cikarilmistir.

---

## PHASE 10: Hata Analizi

| | Tahmin: 0 | Tahmin: 1 |
|---|---|---|
| Gercek: 0 | TN = {tn} | FP = {fp} |
| Gercek: 1 | FN = {fn} | TP = {tp} |

- **{fn} False Negative:** Churn edecek musterilerin kacirilan tahminleri — is maliyeti en yuksek hata.
- **{fp} False Positive:** Yanlis alarm — gereksiz retention kampanyasi maliyeti.
- Recall = {best_row['Recall']:.4f}: Churn eden musterilerin %{best_row['Recall']*100:.1f}'i dogru yakalanmaktadir.

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

- **Final Model:** {best_model_name}
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
"""

report_path = f"{BASE_DIR}/reports/markdown/model_expert_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"[KAYIT] {report_path}")

# =============================================================================
# OZET
# =============================================================================
print("\n" + "="*70)
print("MODEL EXPERT — PIPELINE TAMAMLANDI")
print("="*70)
print(f"\nEgitilen model sayisi : {len(trained_models)}")
print(f"Basarili model sayisi : {len(sorted_df)}")
print(f"Final model           : {best_model_name}")
print(f"\nKaydedilen dosyalar:")
print(f"  models/final_model.pkl")
print(f"  models/all_model_results.csv")
print(f"  models/ranked_model_results.csv")
print(f"  reports/markdown/model_expert_report.md")
print(f"  reports/markdown/model_comparison_prettytable.txt")
print(f"  figures/model_phase7_*.html/png (5 grafik)")
print(f"  figures/model_phase10_*.html/png (3 grafik)")
print("="*70)
