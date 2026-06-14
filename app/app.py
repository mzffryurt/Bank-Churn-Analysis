# =============================================================================
# Churn Tahmin Uygulamasi - Streamlit Deployment
# Model: Logistic Regression (class_weight='balanced')
# Test F1=0.4866 | Recall=0.6716 | ROC-AUC=0.7629
# Shneiderman 8 Altin Kural + Nielsen HCI Prensipleri
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime
import warnings
import io

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SAYFA YAPILANDIRMASI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Musteri Churn Tahmin Sistemi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PROJE YOLLARI (app.py'nin bir ust dizininden referans alinir)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports" / "markdown"
MODELS_DIR = BASE_DIR / "models"
LOG_PATH = BASE_DIR / "logs" / "prediction_log.csv"

# ---------------------------------------------------------------------------
# RENK PALETİ
# ---------------------------------------------------------------------------
PAL = {
    "primary":   "#2E86AB",
    "secondary": "#6A994E",
    "accent":    "#F18F01",
    "danger":    "#C73E1D",
    "purple":    "#8E7DBE",
    "text":      "#1F2937",
    "muted":     "#6B7280",
    "border":    "#D1D5DB",
    "card":      "#F9FAFB",
}

# ---------------------------------------------------------------------------
# CUSTOM CSS  (Shneiderman Kural 1 - Tutarlilik)
# Inter font + profesyonel kurumsal dashboard estetigi
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        /* ---------- FONT: Inter (Minimal Swiss - dashboard/admin icin ideal) ---------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp, .main, button, input, select, textarea {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}

        /* ---------- ANA ARKA PLAN ---------- */
        .stApp {{
            background: #F4F8FB;
        }}
        .main {{
            background: transparent;
        }}
        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }}

        /* ---------- HERO HEADER KARTI ---------- */
        /* Koyu mavi gradyan, beyaz metin, yuvarlak koseler, derin golge */
        .hero-header {{
            background: linear-gradient(135deg, #1A5F7A 0%, #2E86AB 55%, #3A9DC2 100%);
            border-radius: 20px;
            padding: 30px 38px;
            margin-bottom: 26px;
            box-shadow:
                0 10px 40px rgba(46, 134, 171, 0.35),
                0 2px 8px rgba(0, 0, 0, 0.08);
            position: relative;
            overflow: hidden;
        }}
        /* Decorative blur circle - arka plan derinligi */
        .hero-header::before {{
            content: '';
            position: absolute;
            top: -60px;
            right: -60px;
            width: 220px;
            height: 220px;
            background: rgba(255, 255, 255, 0.07);
            border-radius: 50%;
        }}
        .hero-header::after {{
            content: '';
            position: absolute;
            bottom: -40px;
            right: 120px;
            width: 140px;
            height: 140px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50%;
        }}
        .hero-title {{
            color: #FFFFFF;
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0 0 6px 0;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }}
        .hero-subtitle {{
            color: rgba(255, 255, 255, 0.82);
            font-size: 0.95rem;
            font-weight: 400;
            margin: 0;
            line-height: 1.5;
        }}
        .hero-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.28);
            color: #FFFFFF;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-top: 12px;
            letter-spacing: 0.02em;
        }}

        /* ---------- GENEL KART (beyaz, yuvarlak, golge) ---------- */
        .hero-card {{
            background: #FFFFFF;
            border: 1px solid {PAL['border']};
            border-radius: 16px;
            padding: 24px 28px;
            box-shadow: 0 4px 20px rgba(31, 41, 55, 0.07),
                        0 1px 4px rgba(31, 41, 55, 0.04);
            margin-bottom: 20px;
        }}

        /* ---------- METRIK / KPI KARTI ---------- */
        .metric-card {{
            background: #FFFFFF;
            border: 1px solid {PAL['border']};
            border-radius: 14px;
            padding: 20px 18px;
            box-shadow: 0 4px 16px rgba(31, 41, 55, 0.07),
                        0 1px 3px rgba(31, 41, 55, 0.04);
            text-align: center;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
            cursor: default;
        }}
        .metric-card:hover {{
            box-shadow: 0 8px 28px rgba(31, 41, 55, 0.12);
            transform: translateY(-2px);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {PAL['primary']};
            line-height: 1.1;
            letter-spacing: -0.02em;
        }}
        .metric-label {{
            font-size: 0.80rem;
            color: {PAL['muted']};
            margin-top: 6px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        /* ---------- TAHMIN SONUC KARTLARI ---------- */
        .result-positive {{
            background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
            border-radius: 16px;
            padding: 24px 28px;
            border: 1.5px solid #A7F3D0;
            box-shadow: 0 4px 16px rgba(106, 153, 78, 0.10);
        }}
        .result-warning {{
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border-radius: 16px;
            padding: 24px 28px;
            border: 1.5px solid #FDE68A;
            box-shadow: 0 4px 16px rgba(241, 143, 1, 0.10);
        }}
        .result-danger {{
            background: linear-gradient(135deg, #FFF1F0 0%, #FFE4E1 100%);
            border-radius: 16px;
            padding: 24px 28px;
            border: 1.5px solid #FECACA;
            box-shadow: 0 4px 16px rgba(199, 62, 29, 0.10);
        }}

        /* ---------- TAHMIN SONUCU BUYUK BASLIK ---------- */
        .result-label {{
            font-size: 1.55rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }}
        .result-icon-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .result-prob-row {{
            font-size: 1.05rem;
            font-weight: 500;
            margin: 0 0 10px 0;
        }}
        .result-disclaimer {{
            font-size: 0.82rem;
            color: {PAL['muted']};
            margin-top: 10px;
            border-top: 1px solid rgba(0,0,0,0.08);
            padding-top: 10px;
        }}

        /* ---------- RISK ETİKETLERİ ---------- */
        .tag-low  {{
            background: #D1FAE5; color: #065F46;
            border-radius: 20px; padding: 3px 14px;
            font-size: 0.80rem; font-weight: 700;
            display: inline-block;
        }}
        .tag-mid  {{
            background: #FEF3C7; color: #92400E;
            border-radius: 20px; padding: 3px 14px;
            font-size: 0.80rem; font-weight: 700;
            display: inline-block;
        }}
        .tag-high {{
            background: #FFE4E1; color: #7F1D1D;
            border-radius: 20px; padding: 3px 14px;
            font-size: 0.80rem; font-weight: 700;
            display: inline-block;
        }}

        /* ---------- TİPOGRAFİ HİYERARŞİSİ ---------- */
        /* h1: Sayfa basliği seviyesi */
        h1 {{
            color: {PAL['text']};
            font-size: 1.80rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            line-height: 1.25;
        }}
        /* h2: Sekme / bolum basliği */
        h2 {{
            color: {PAL['text']};
            font-size: 1.30rem;
            font-weight: 700;
            letter-spacing: -0.015em;
            line-height: 1.3;
        }}
        /* h3: Alt bolum / kart basliği */
        h3 {{
            color: {PAL['text']};
            font-size: 1.02rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            line-height: 1.4;
        }}

        /* ---------- SEKME STİLİ ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background: #EEF3F7;
            padding: 4px 6px;
            border-radius: 12px;
            border: 1px solid {PAL['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 9px;
            padding: 7px 16px;
            font-size: 0.88rem;
            font-weight: 500;
            color: {PAL['muted']};
            transition: all 0.18s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PAL['primary']};
            color: #FFFFFF !important;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(46, 134, 171, 0.28);
        }}

        /* ---------- FORM ELEMANLARI ---------- */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            border-radius: 10px !important;
            border: 1.5px solid {PAL['border']} !important;
            font-size: 0.92rem !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {PAL['primary']} !important;
            box-shadow: 0 0 0 3px rgba(46, 134, 171, 0.14) !important;
        }}

        /* ---------- BUTON ---------- */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.90rem;
            transition: all 0.18s ease;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #2E86AB 0%, #1A6B8A 100%);
            border: none;
            box-shadow: 0 4px 14px rgba(46, 134, 171, 0.30);
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 6px 20px rgba(46, 134, 171, 0.42);
            transform: translateY(-1px);
        }}

        /* ---------- SIDEBAR ---------- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1A2535 0%, #243447 50%, #2E3D52 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}
        [data-testid="stSidebar"] * {{
            color: #E2E8F0 !important;
        }}
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li {{
            color: #94A3B8 !important;
            font-size: 0.84rem !important;
        }}
        [data-testid="stSidebar"] h3 {{
            color: #F1F5F9 !important;
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.10) !important;
        }}

        /* ---------- YARDIMCI SINIFLAR ---------- */
        .small-muted {{
            color: {PAL['muted']};
            font-size: 0.87rem;
            line-height: 1.6;
        }}
        .divider {{
            border: none;
            border-top: 1px solid {PAL['border']};
            margin: 18px 0;
        }}
        .section-header {{
            font-size: 0.78rem;
            font-weight: 700;
            color: {PAL['muted']};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 20px 0 10px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid {PAL['border']};
        }}

        /* ---------- DATAFRAME / TABLO ---------- */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {PAL['border']};
        }}

        /* ---------- REDUCED MOTION (erisim) ---------- */
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MODEL YUKLEME  (Shneiderman Kural 3 - Bilgilendirici geri bildirim)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yukleniyor...")
def load_assets():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


# ---------------------------------------------------------------------------
# YARDIMCI: Girdi DataFrame olusturma
# Ham 10 alan + 2 turetilmis ozellik
# ---------------------------------------------------------------------------
PIPELINE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "is_high_products_risk", "has_zero_balance",
]

EXPECTED_RAW_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary",
]

MODEL_FEATURE_NAMES = [
    "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain", "Gender_Male",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "is_high_products_risk", "has_zero_balance",
]

GEOGRAPHY_OPTIONS = ["France", "Spain", "Germany"]
GENDER_OPTIONS    = ["Male", "Female"]


def build_input_df(
    credit_score, geography, gender, age, tenure,
    balance, num_products, has_cr_card, is_active, salary,
):
    """Ham 10 girdiyi pipeline'in bekledigine donustur (12 sutun)."""
    is_high = int(num_products >= 3)
    has_zero = int(balance == 0)
    return pd.DataFrame(
        [{
            "CreditScore":          credit_score,
            "Geography":            geography,
            "Gender":               gender,
            "Age":                  age,
            "Tenure":               tenure,
            "Balance":              balance,
            "NumOfProducts":        num_products,
            "HasCrCard":            has_cr_card,
            "IsActiveMember":       is_active,
            "EstimatedSalary":      salary,
            "is_high_products_risk": is_high,
            "has_zero_balance":     has_zero,
        }],
        columns=PIPELINE_COLS,
    )


def predict(input_df, model, pipeline):
    """Pipeline transform + model predict. (numpy array gecer, uyari supressed)"""
    X = pipeline.transform(input_df)
    X_named = pd.DataFrame(X, columns=MODEL_FEATURE_NAMES)
    pred  = model.predict(X_named)[0]
    proba = model.predict_proba(X_named)[0]
    return int(pred), float(proba[1])   # sinif, churn olasiligi


def risk_level(churn_prob: float):
    if churn_prob < 0.40:
        return "Dusuk", "tag-low",  "result-positive", "#6A994E"
    elif churn_prob < 0.65:
        return "Orta",  "tag-mid",  "result-warning",  "#F18F01"
    else:
        return "Yuksek","tag-high", "result-danger",   "#C73E1D"


# ---------------------------------------------------------------------------
# LOGLAMA  (Monitoring altyapisi)
# ---------------------------------------------------------------------------
def log_prediction(source: str, input_df: pd.DataFrame, pred: int, churn_prob: float):
    row = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "source":      source,
        "prediction":  pred,
        "churn_prob":  round(churn_prob, 4),
    }
    for col in EXPECTED_RAW_COLS:
        if col in input_df.columns:
            row[col] = input_df.iloc[0][col]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame([row])
    if LOG_PATH.exists():
        log_df.to_csv(LOG_PATH, mode="a", index=False, header=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)


# ---------------------------------------------------------------------------
# GAUGE GRAFİGİ
# ---------------------------------------------------------------------------
def gauge_chart(churn_prob: float, title: str = "Churn Olasiligi"):
    color = "#6A994E" if churn_prob < 0.40 else ("#F18F01" if churn_prob < 0.65 else "#C73E1D")
    # Arka plan track rengi: risk seviyesine gore hafif ton
    track_color = "#F0FFF4" if churn_prob < 0.40 else ("#FFFBEB" if churn_prob < 0.65 else "#FFF1F0")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(churn_prob * 100, 1),
        delta={
            "reference": 40,
            "increasing": {"color": "#C73E1D"},
            "decreasing": {"color": "#6A994E"},
            "valueformat": ".1f",
        },
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 14, "color": "#1F2937", "family": "Inter, sans-serif"},
        },
        number={
            "suffix": "%",
            "font": {"size": 32, "color": color, "family": "Inter, sans-serif"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#D1D5DB",
                "tickfont": {"size": 10, "color": "#9CA3AF"},
                "nticks": 6,
            },
            "bar":  {"color": color, "thickness": 0.30},
            "bgcolor": track_color,
            "borderwidth": 1.5,
            "bordercolor": "#E5E7EB",
            "steps": [
                {"range": [0,  40], "color": "#ECFDF5"},
                {"range": [40, 65], "color": "#FFFBEB"},
                {"range": [65,100], "color": "#FFF1F0"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": round(churn_prob * 100, 1),
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin={"t": 50, "b": 15, "l": 25, "r": 25},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR  (Shneiderman Kural 7 - Kullaniciya kontrol hissi,
#           Kural 8 - Bellek yukunu azaltma)
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Marka basligi
        st.markdown(
            """
            <div style='padding:8px 0 16px 0'>
                <div style='font-size:1.05rem;font-weight:800;color:#F1F5F9;letter-spacing:-0.01em'>
                    Churn Tahmin Sistemi
                </div>
                <div style='font-size:0.76rem;color:#64748B;margin-top:3px;font-weight:500;
                            text-transform:uppercase;letter-spacing:0.06em'>
                    Logistic Regression &bull; v1.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Model performans metrikleri
        st.markdown("### Model Ozeti")
        sidebar_metrics = [
            ("F1 (test)",  "0.4866", "#60A5FA"),
            ("Recall",     "0.6716", "#34D399"),
            ("ROC-AUC",    "0.7629", "#FBBF24"),
            ("Accuracy",   "0.7130", "#A78BFA"),
            ("Overfit",    "0.0103", "#F87171"),
        ]
        for label, val, color in sidebar_metrics:
            st.markdown(
                f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                    <span style='font-size:0.82rem;color:#94A3B8'>{label}</span>
                    <span style='font-size:0.88rem;font-weight:700;color:{color}'>{val}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Risk seviyeleri aciklamasi
        st.markdown("### Risk Seviyeleri")
        st.markdown(
            """
            <div style='display:flex;flex-direction:column;gap:7px;margin-top:4px'>
                <div style='display:flex;align-items:center;justify-content:space-between'>
                    <span style='background:#065F46;color:#D1FAE5;border-radius:8px;
                                 padding:2px 10px;font-size:0.78rem;font-weight:700'>Dusuk</span>
                    <span style='font-size:0.80rem;color:#94A3B8'>&lt; %40</span>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between'>
                    <span style='background:#92400E;color:#FEF3C7;border-radius:8px;
                                 padding:2px 10px;font-size:0.78rem;font-weight:700'>Orta</span>
                    <span style='font-size:0.80rem;color:#94A3B8'>%40 – %65</span>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between'>
                    <span style='background:#7F1D1D;color:#FFE4E1;border-radius:8px;
                                 padding:2px 10px;font-size:0.78rem;font-weight:700'>Yuksek</span>
                    <span style='font-size:0.80rem;color:#94A3B8'>&gt; %65</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Sistem bilgisi
        st.markdown(
            """
            <div style='font-size:0.75rem;color:#475569;line-height:1.7;padding-bottom:4px'>
                Deployment Expert<br>
                Churn Analysis Pipeline<br>
                <span style='color:#334155'>Shneiderman 8 Altin Kural</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# SEKME 1: TEKIL TAHMİN
# ---------------------------------------------------------------------------
def tab_single(model, pipeline):
    st.markdown("## Tekil Musteri Churn Tahmini")
    st.markdown(
        "<p class='small-muted'>Musteri bilgilerini girin; model churn olasiligi, risk seviyesi ve eylem onerisi hesaplasin.</p>",
        unsafe_allow_html=True,
    )

    # --- Ornek veri ile doldurma (Shneiderman Kural 2 - Kisayollar) ---
    col_hint1, col_hint2, col_hint3 = st.columns([2, 2, 4])
    with col_hint1:
        if st.button("Ornek: Dusuk Riskli Musteri", use_container_width=True):
            st.session_state.update({
                "cs": 720, "geo": "France", "gen": "Male",
                "age": 35, "ten": 7, "bal": 85000.0,
                "nop": 2, "hcc": 1, "iam": 1, "sal": 60000.0,
            })
    with col_hint2:
        if st.button("Ornek: Yuksek Riskli Musteri", use_container_width=True):
            st.session_state.update({
                "cs": 480, "geo": "Germany", "gen": "Female",
                "age": 52, "ten": 1, "bal": 0.0,
                "nop": 3, "hcc": 0, "iam": 0, "sal": 40000.0,
            })
    with col_hint3:
        if st.button("Formu Sifirla", use_container_width=False):
            for k in ["cs","geo","gen","age","ten","bal","nop","hcc","iam","sal","prediction_done"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # --- Form girisleri (Shneiderman Kural 5 - Hata onleme) ---
    with st.form("single_pred_form", clear_on_submit=False):

        # GRUP 1: Demografik bilgiler
        st.markdown("<div class='section-header'>Demografik Bilgiler</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            geography = st.selectbox(
                "Ulke / Cografi Bolge",
                GEOGRAPHY_OPTIONS,
                index=GEOGRAPHY_OPTIONS.index(st.session_state.get("geo", "France")),
                help="Musterinin kayitli oldugu ulke",
            )
        with c2:
            gender = st.selectbox(
                "Cinsiyet",
                GENDER_OPTIONS,
                index=GENDER_OPTIONS.index(st.session_state.get("gen", "Male")),
                help="Musterinin cinsiyeti",
            )
        with c3:
            age = st.number_input(
                "Yas",
                min_value=18, max_value=95,
                value=int(st.session_state.get("age", 40)),
                step=1,
                help="Musterinin yasi (18-95 arasi)",
            )

        # GRUP 2: Finansal profil
        st.markdown("<div class='section-header'>Finansal Profil</div>", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            credit_score = st.number_input(
                "Kredi Skoru",
                min_value=300, max_value=850,
                value=int(st.session_state.get("cs", 650)),
                step=1,
                help="300-850 arasi kredi skoru. Dusuk skor risk gostergesi.",
            )
        with c5:
            balance = st.number_input(
                "Hesap Bakiyesi (USD)",
                min_value=0.0, max_value=300000.0,
                value=float(st.session_state.get("bal", 50000.0)),
                step=100.0,
                help="Musterinin banka hesap bakiyesi. Sifir bakiye risk sinyali.",
            )
        with c6:
            salary = st.number_input(
                "Tahmini Maas (USD/yil)",
                min_value=0.0, max_value=500000.0,
                value=float(st.session_state.get("sal", 50000.0)),
                step=500.0,
                help="Musterinin yillik tahmini maasi",
            )

        # GRUP 3: Urun ve iliski bilgisi
        st.markdown("<div class='section-header'>Urun ve Iliski Bilgisi</div>", unsafe_allow_html=True)
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            tenure = st.number_input(
                "Hizmet Suresi (yil)",
                min_value=0, max_value=10,
                value=int(st.session_state.get("ten", 5)),
                step=1,
                help="Musterinin bankadaki yil sayisi (0-10)",
            )
        with c8:
            num_products = st.number_input(
                "Urun Sayisi",
                min_value=1, max_value=4,
                value=int(st.session_state.get("nop", 2)),
                step=1,
                help="Musterinin sahip oldugu urun sayisi (1-4). 3+ yuksek risk faktoru.",
            )
        with c9:
            has_cr_card = st.selectbox(
                "Kredi Karti Var mi?",
                options=[1, 0],
                format_func=lambda x: "Evet" if x == 1 else "Hayir",
                index=0 if st.session_state.get("hcc", 1) == 1 else 1,
                help="Musterinin kredi karti sahipligi",
            )
        with c10:
            is_active = st.selectbox(
                "Aktif Uye mi?",
                options=[1, 0],
                format_func=lambda x: "Evet" if x == 1 else "Hayir",
                index=0 if st.session_state.get("iam", 1) == 1 else 1,
                help="Musterinin son donemde aktif olup olmadigi",
            )

        # Turetilmis ozellik bilgilendirmesi
        if num_products >= 3:
            st.info("Urun sayisi 3 veya uzerinde: **yuksek urun riski** aktif (is_high_products_risk=1)")
        if balance == 0:
            st.warning("Hesap bakiyesi sifir: **sifir bakiye riski** aktif (has_zero_balance=1)")

        submitted = st.form_submit_button(
            "Churn Tahmini Yap",
            type="primary",
            use_container_width=True,
        )

    # --- Tahmin (Shneiderman Kural 4 - Diyalogun kapanisi) ---
    if submitted:
        with st.spinner("Model hesapliyor..."):
            input_df = build_input_df(
                credit_score, geography, gender, age, tenure,
                balance, num_products, has_cr_card, is_active, salary,
            )
            pred, churn_prob = predict(input_df, model, pipeline)
            st.session_state["last_prediction"] = {"pred": pred, "prob": churn_prob, "df": input_df}
            log_prediction("single", input_df, pred, churn_prob)
            st.session_state["prediction_done"] = True

    # Sonuclari goster (session state ile -- Kural 6 geri alinabilirlik)
    if st.session_state.get("prediction_done") and "last_prediction" in st.session_state:
        lp = st.session_state["last_prediction"]
        pred, churn_prob = lp["pred"], lp["prob"]
        risk, tag_cls, card_cls, risk_color = risk_level(churn_prob)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### Tahmin Sonucu")

        col_res, col_gauge = st.columns([3, 2])

        with col_res:
            label    = "CHURN EDECEK" if pred == 1 else "CHURN ETMEYECEK"
            subtitle = "Musteri kaybedilebilir — aksiyon gerekli." if pred == 1 else "Musteri elde tutulabilir."
            disclaimer = (
                "Bu sonuc makine ogrenmesi modelinin tahminidir. "
                "Kritik kararlarda uzman degerlendirmesiyle birlikte kullaniniz."
            )
            # Sonuc kartinin arka plan rengi risk seviyesine gore degisiyor
            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div class="result-label" style="color:{risk_color}">{label}</div>
                    <div class="result-icon-row">
                        <span class="{tag_cls}">Risk: {risk}</span>
                        <span style="font-size:0.90rem;color:#374151">{subtitle}</span>
                    </div>
                    <div class="result-prob-row">
                        Churn Olasiligi:
                        <b style="color:{risk_color};font-size:1.35rem">{churn_prob*100:.1f}%</b>
                    </div>
                    <div style="display:flex;gap:20px;margin-bottom:6px">
                        <div style="text-align:center">
                            <div style="font-size:1.15rem;font-weight:700;color:#C73E1D">{churn_prob*100:.1f}%</div>
                            <div style="font-size:0.76rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.04em">Churn Olasiligi</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:1.15rem;font-weight:700;color:#6A994E">{(1-churn_prob)*100:.1f}%</div>
                            <div style="font-size:0.76rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.04em">Kalma Olasiligi</div>
                        </div>
                    </div>
                    <div class="result-disclaimer">{disclaimer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Oneriler (Shneiderman Kural 3 - Bilgilendirici geri bildirim)
            st.markdown("#### Eylem Onerisi")
            if pred == 1 and churn_prob >= 0.65:
                st.error(
                    "ACIL: Bu musteri yuksek churn riskinde. "
                    "Kisisellestirilmis retention kampanyasi ve ozel teklif onerilir."
                )
            elif pred == 1 and churn_prob >= 0.40:
                st.warning(
                    "DIKKAT: Orta risk. Musteriye check-in yapilmasi ve "
                    "ek urun teklifinin degerlendirilmesi onerilir."
                )
            else:
                st.success(
                    "IYI: Musteri dusuk riskli. Mevcut iliskiyi surdurmek yeterlidir."
                )

        with col_gauge:
            st.plotly_chart(
                gauge_chart(churn_prob, "Churn Olasiligi"),
                use_container_width=True,
            )

            # Alternatif olasiliklar
            fig_bar = go.Figure(go.Bar(
                x=["Churn Etmeyecek", "Churn Edecek"],
                y=[(1 - churn_prob) * 100, churn_prob * 100],
                marker_color=["#6A994E", "#C73E1D"],
                text=[f"{(1-churn_prob)*100:.1f}%", f"{churn_prob*100:.1f}%"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title={"text": "Sinif Olasiliklari", "font": {"size": 13}},
                yaxis={"title": "%", "range": [0, 110]},
                xaxis={"title": ""},
                height=220,
                margin={"t": 40, "b": 10, "l": 10, "r": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# SEKME 2: TOPLU TAHMİN (CSV)
# ---------------------------------------------------------------------------
def tab_batch(model, pipeline):
    st.markdown("## Toplu Churn Tahmini (CSV)")
    st.markdown(
        "<p class='small-muted'>CSV dosyasi yukleyin; tum satirlar icin tahmin yapilsin, risk dagilimi gorsellestirilsin ve sonuclar indirilebilir tablo olarak sunulsun.</p>",
        unsafe_allow_html=True,
    )

    # Beklenen kolon seti
    st.info(
        f"Beklenen sutunlar: **{', '.join(EXPECTED_RAW_COLS)}**  "
        "*(RowNumber, CustomerId, Surname gibi kimlik sutunlari gerekli degil; varsa otomatik dislenir.)*"
    )

    # Ornek CSV indirme (Shneiderman Kural 2 - Kisayollar)
    sample_data = pd.DataFrame([
        {
            "CreditScore": 720, "Geography": "France",  "Gender": "Male",
            "Age": 35, "Tenure": 7, "Balance": 85000.0,
            "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000.0,
        },
        {
            "CreditScore": 480, "Geography": "Germany", "Gender": "Female",
            "Age": 52, "Tenure": 1, "Balance": 0.0,
            "NumOfProducts": 3, "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 40000.0,
        },
    ])
    sample_csv = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Ornek CSV Indir (sifreli)",
        data=sample_csv,
        file_name="ornek_musteri.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "CSV Dosyasi Yukle",
        type=["csv"],
        help="Maksimum onerilen boyut: 10 MB, maksimum 50 000 satir",
    )

    if uploaded is None:
        st.markdown(
            "<p class='small-muted'>Henuz dosya yuklenmedi. Yukle butonunu kullanarak CSV secin.</p>",
            unsafe_allow_html=True,
        )
        return

    # --- CSV dogrulama (Shneiderman Kural 5 - Hata onleme) ---
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV okunamadi: {e}")
        return

    st.markdown(f"**{len(df_raw):,} satir, {len(df_raw.columns)} sutun** yuklendi.")

    # Kimlik sutunlarini disla
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df_raw.columns]
    if drop_cols:
        st.info(f"Kimlik sutunlari otomatik cikarildi: {drop_cols}")
        df_raw = df_raw.drop(columns=drop_cols)

    # Eksik sutun kontrolu
    missing_cols = [c for c in EXPECTED_RAW_COLS if c not in df_raw.columns]
    if missing_cols:
        st.error(
            f"Eksik sutunlar tespit edildi: **{missing_cols}**  \n"
            "Lutfen dosyanizi yukardaki beklenen sutun listesine gore duzenleyin."
        )
        return

    # Fazla sutunlar
    extra_cols = [c for c in df_raw.columns if c not in EXPECTED_RAW_COLS]
    if extra_cols:
        st.warning(f"Fazladan sutunlar yoksayildi: {extra_cols}")
        df_raw = df_raw[EXPECTED_RAW_COLS]

    # Veri tipi kontrolu
    type_errors = []
    for col in ["CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]:
        try:
            df_raw[col] = df_raw[col].astype(int)
        except Exception:
            type_errors.append(col)
    for col in ["Balance", "EstimatedSalary"]:
        try:
            df_raw[col] = df_raw[col].astype(float)
        except Exception:
            type_errors.append(col)
    if type_errors:
        st.error(f"Veri tipi donusturme hatasi: {type_errors}. Sayisal sutunlari kontrol edin.")
        return

    geo_valid   = set(GEOGRAPHY_OPTIONS)
    gen_valid   = set(GENDER_OPTIONS)
    bad_geo = df_raw[~df_raw["Geography"].isin(geo_valid)]
    bad_gen = df_raw[~df_raw["Gender"].isin(gen_valid)]
    if len(bad_geo) > 0:
        st.error(f"{len(bad_geo)} satirda gecersiz 'Geography' degeri. Kabul edilenler: {GEOGRAPHY_OPTIONS}")
        return
    if len(bad_gen) > 0:
        st.error(f"{len(bad_gen)} satirda gecersiz 'Gender' degeri. Kabul edilenler: {GENDER_OPTIONS}")
        return

    # Eksik deger kontrolu
    null_counts = df_raw.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        st.warning(f"Eksik degerler: {null_cols.to_dict()}. Bu satirlar atlanabilir.")
        df_raw = df_raw.dropna()
        st.info(f"Eksik degerli satirlar kaldirildi. Kalan: {len(df_raw):,} satir.")

    if len(df_raw) == 0:
        st.error("Gecerli veri satiri kalmadi.")
        return

    # --- Toplu tahmin ---
    if st.button("Tum Satirlari Tahmin Et", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Tahminler hesaplaniyor...")
        results = []
        batch_size = max(1, len(df_raw) // 20)

        for i in range(0, len(df_raw), batch_size):
            chunk = df_raw.iloc[i:i + batch_size].copy()
            chunk["is_high_products_risk"] = (chunk["NumOfProducts"] >= 3).astype(int)
            chunk["has_zero_balance"]      = (chunk["Balance"] == 0).astype(int)
            chunk = chunk[PIPELINE_COLS]
            X = pipeline.transform(chunk)
            X_named = pd.DataFrame(X, columns=MODEL_FEATURE_NAMES)
            preds  = model.predict(X_named)
            probas = model.predict_proba(X_named)[:, 1]
            for j in range(len(chunk)):
                results.append({
                    "Churn_Tahmini":  int(preds[j]),
                    "Churn_Olasiligi": round(float(probas[j]) * 100, 2),
                    "Risk_Seviyesi":   risk_level(float(probas[j]))[0],
                })
            progress_bar.progress(
                min((i + batch_size) / len(df_raw), 1.0),
                text=f"{min(i+batch_size, len(df_raw)):,}/{len(df_raw):,} satir islendi",
            )

        progress_bar.empty()
        result_df = df_raw.reset_index(drop=True).copy()
        result_df["Churn_Tahmini"]   = [r["Churn_Tahmini"]   for r in results]
        result_df["Churn_Olasiligi"] = [r["Churn_Olasiligi"] for r in results]
        result_df["Risk_Seviyesi"]   = [r["Risk_Seviyesi"]   for r in results]

        st.session_state["batch_result"] = result_df
        st.success(f"{len(result_df):,} musteri icin tahmin tamamlandi.")

    # Sonuclari goster
    if "batch_result" in st.session_state:
        res = st.session_state["batch_result"]

        # Ozet istatistikler (Shneiderman Kural 4 - Diyalogun kapanisi)
        n_total  = len(res)
        n_churn  = (res["Churn_Tahmini"] == 1).sum()
        n_retain = n_total - n_churn
        avg_prob = res["Churn_Olasiligi"].mean()

        st.markdown("<div class='section-header'>Ozet Istatistikler</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{n_total:,}</div>"
                "<div class='metric-label'>Toplam Musteri</div></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='color:#C73E1D'>{n_churn:,}</div>"
                "<div class='metric-label'>Churn Tahmini</div></div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='color:#6A994E'>{n_retain:,}</div>"
                "<div class='metric-label'>Elde Tutulabilir</div></div>",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='color:#F18F01'>{avg_prob:.1f}%</div>"
                "<div class='metric-label'>Ort. Churn Olasiligi</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk dagilimi grafigi
        risk_counts = res["Risk_Seviyesi"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Sayi"]
        color_map = {"Dusuk": "#6A994E", "Orta": "#F18F01", "Yuksek": "#C73E1D"}
        fig_risk = px.bar(
            risk_counts, x="Risk", y="Sayi", color="Risk",
            color_discrete_map=color_map,
            text="Sayi",
            title="Risk Seviyesi Dagilimi",
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(
            height=300, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 50, "b": 20, "l": 20, "r": 20},
        )

        # Olasilik dagilimi histogrami
        fig_hist = px.histogram(
            res, x="Churn_Olasiligi", nbins=30,
            title="Churn Olasiligi Dagilimi",
            color_discrete_sequence=["#2E86AB"],
        )
        fig_hist.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 50, "b": 20, "l": 20, "r": 20},
            xaxis_title="Churn Olasiligi (%)",
            yaxis_title="Musteri Sayisi",
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(fig_risk, use_container_width=True)
        with col_g2:
            st.plotly_chart(fig_hist, use_container_width=True)

        # Tablo ve indirme (Shneiderman Kural 3 - Bilgilendirici geri bildirim)
        st.markdown("<div class='section-header'>Sonuc Tablosu</div>", unsafe_allow_html=True)
        st.dataframe(
            res.sort_values("Churn_Olasiligi", ascending=False).head(200),
            use_container_width=True,
            height=360,
        )

        csv_out = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Tum Sonuclari CSV Olarak Indir",
            data=csv_out,
            file_name=f"churn_tahmin_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
        )

        # Loglama
        if st.button("Toplu Tahmini Logla"):
            for idx in range(min(len(res), 5000)):
                row = res.iloc[[idx]][EXPECTED_RAW_COLS]
                log_prediction("batch", row, int(res.iloc[idx]["Churn_Tahmini"]), res.iloc[idx]["Churn_Olasiligi"] / 100)
            st.success("Toplu tahmin log dosyasina kaydedildi.")

        # Temizleme (Shneiderman Kural 6 - Geri alinabilirlik)
        if st.button("Toplu Tahmin Sonuclarini Temizle"):
            del st.session_state["batch_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# SEKME 3: MODEL PERFORMANSI
# ---------------------------------------------------------------------------
def tab_performance():
    st.markdown("## Model Performans Paneli")
    st.markdown(
        "<p class='small-muted'>Final model ve karskilastirma grafikleri, confusion matrix ve ROC egrisi.</p>",
        unsafe_allow_html=True,
    )

    # Model karsilastirma tablosu
    ranked_path = MODELS_DIR / "ranked_model_results.csv"
    if ranked_path.exists():
        df_ranked = pd.read_csv(ranked_path)
        st.markdown("### Model Karsilastirma Tablosu")
        cols_show = ["Model", "Test F1", "F1", "Recall", "Precision", "ROC-AUC", "Accuracy", "Overfit Gap", "Overfit", "MultiCriteria", "Skor"]
        available_cols = [c for c in cols_show if c in df_ranked.columns]
        
        max_subset = [c for c in ["Test F1", "F1", "ROC-AUC", "Recall", "MultiCriteria", "Skor"] if c in available_cols]
        min_subset = [c for c in ["Overfit Gap", "Overfit"] if c in available_cols]
        
        st.dataframe(
            df_ranked[available_cols].style.highlight_max(
                subset=max_subset,
                color="#D5F5E3",
            ).highlight_min(
                subset=min_subset,
                color="#D5F5E3",
            ),
            use_container_width=True,
            height=400,
        )
    else:
        st.warning("ranked_model_results.csv bulunamadi.")

    # HTML grafikleri
    figure_configs = [
        ("model_phase10_final_confusion_matrix.html", "Final Confusion Matrix", 520),
        ("model_phase10_roc_all_models.html",          "ROC Egrisi (Tum Modeller)", 520),
        ("model_phase10_precision_recall_curve.html",  "Precision-Recall Egrisi", 520),
        ("model_phase7_performance_comparison.html",   "Performans Karsilastirmasi", 540),
        ("model_phase7_cv_stability.html",             "CV Kararliligi", 520),
        ("model_phase7_overfitting_analysis.html",     "Overfitting Analizi", 520),
        ("model_phase7_leadership_matrix.html",        "Liderlik Matrisi", 540),
        ("model_phase7_training_time.html",            "Egitim Suresi", 400),
    ]

    for filename, title, height in figure_configs:
        fig_path = FIGURES_DIR / filename
        if fig_path.exists():
            st.markdown(f"### {title}")
            with open(fig_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=height, scrolling=True)
        else:
            png_path = FIGURES_DIR / filename.replace(".html", ".png")
            if png_path.exists():
                st.markdown(f"### {title}")
                st.image(str(png_path), use_column_width=True)


# ---------------------------------------------------------------------------
# SEKME 4: MODEL BİLGİSİ
# ---------------------------------------------------------------------------
def tab_model_info():
    st.markdown("## Model Bilgisi ve Secim Gerekceleri")
    st.markdown(
        "<p class='small-muted'>Final model secim gerekceleri, pipeline ozellikleri ve model sinirliliklar.</p>",
        unsafe_allow_html=True,
    )

    # Metrik ozet kartlar
    st.markdown("<div class='section-header'>Performans Metrikleri</div>", unsafe_allow_html=True)
    metrics = [
        ("Test F1 (Pozitif)", "0.4866", PAL["primary"]),
        ("Recall",            "0.6716", PAL["secondary"]),
        ("ROC-AUC",           "0.7629", PAL["accent"]),
        ("Accuracy",          "0.7130", PAL["purple"]),
        ("Overfit Gap",       "0.0103", PAL["danger"]),
        ("CV F1 Std",         "0.0246", PAL["muted"]),
    ]
    cols = st.columns(len(metrics))
    for col, (label, val, color) in zip(cols, metrics):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='color:{color}'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_info, col_feat = st.columns([3, 2])

    with col_info:
        st.markdown("### Final Model: Logistic Regression")
        st.markdown("""
**Parametre:** `class_weight='balanced'`, `max_iter=1000`

**Secim Gerekceleri:**
- En yuksek MultiCriteria puani: **0.6381**
- Cok dusuk overfit gap: **0.0103** (uretim kararliligi)
- En yuksek Recall: **0.6716** (churn yakalama odagi)
- CV F1 std deger: **0.0246** (stabil)
- Egitim suresi: **3.73 sn** (urun ortaminda hizli)

**Sinif Dengesizligi:**
Veri seti yaklasik %80/%20 dengesiz. `class_weight='balanced'`
ile churn azinlik sinifina agirlik verildi; SMOTE uygulanmadi.

**Sinirliliklar:**
- Precision degeri dusuk (**0.3815**): model hassas (false positive fazla)
- Bu churn tanima odakli bir model icin bilerek kabul edilmis bir denge
- Oneri: cok dusuk guven skorlu tahminlerde ek dogrulama yapilmasi
        """)

    with col_feat:
        st.markdown("### Pipeline Ozellikleri")
        feature_data = pd.DataFrame({
            "Ozellik":     ["CreditScore","Age","Tenure","Balance","EstimatedSalary",
                            "Geography","Gender","NumOfProducts","HasCrCard",
                            "IsActiveMember","is_high_products_risk","has_zero_balance"],
            "Donusum":     ["StandardScaler","StandardScaler","StandardScaler",
                            "StandardScaler","StandardScaler",
                            "OneHotEncoder","OneHotEncoder",
                            "Passthrough","Passthrough",
                            "Passthrough","Turetilmis","Turetilmis"],
        })
        st.dataframe(feature_data, use_container_width=True, hide_index=True)

    # Rapor
    report_path = REPORTS_DIR / "model_expert_report.md"
    if report_path.exists():
        with st.expander("Model Expert Raporunu Goruntule (genislet)"):
            st.markdown(report_path.read_text(encoding="utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# SEKME 5: MONİTORİNG
# ---------------------------------------------------------------------------
def tab_monitoring():
    st.markdown("## Monitoring ve Loglama")
    st.markdown(
        "<p class='small-muted'>Tahmin gecmisi, guven dagilimi ve model versiyon bilgisi.</p>",
        unsafe_allow_html=True,
    )

    # Model versiyonu bilgisi
    st.markdown("<div class='section-header'>Sistem Durumu</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            "<div class='metric-card'><div class='metric-value' style='color:#2E86AB'>v1.0</div>"
            "<div class='metric-label'>Model Versiyonu</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        model_mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%Y-%m-%d") if MODEL_PATH.exists() else "N/A"
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:#6A994E;font-size:1.2rem'>{model_mtime}</div>"
            "<div class='metric-label'>Model Guncelleme Tarihi</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        log_count = 0
        if LOG_PATH.exists():
            try:
                log_df = pd.read_csv(LOG_PATH)
                log_count = len(log_df)
            except Exception:
                log_count = 0
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:#F18F01'>{log_count:,}</div>"
            "<div class='metric-label'>Toplam Loglanan Tahmin</div></div>",
            unsafe_allow_html=True,
        )

    # Log tablosu
    if LOG_PATH.exists():
        try:
            log_df = pd.read_csv(LOG_PATH)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Son Tahminler")
            st.dataframe(log_df.tail(50).iloc[::-1], use_container_width=True, height=320)

            # Guven dagilimi
            if "churn_prob" in log_df.columns and len(log_df) > 5:
                st.markdown("### Confidence Dagilimi (Tum Tahminler)")
                fig_conf = px.histogram(
                    log_df, x="churn_prob",
                    nbins=20,
                    color_discrete_sequence=["#2E86AB"],
                    title="Churn Olasiligi Dagilimi (Log Gecmisi)",
                )
                fig_conf.update_layout(
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin={"t": 50, "b": 20, "l": 20, "r": 20},
                    xaxis_title="Churn Olasiligi",
                    yaxis_title="Tahmin Sayisi",
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # Log indirme
            st.download_button(
                "Log Dosyasini Indir",
                data=log_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_log.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"Log dosyasi okunamadi: {e}")
    else:
        st.info("Henuz tahmin logu mevcut degil. Tekil veya toplu tahmin yapildikca buraya kaydedilecek.")

    # Input drift uyarisi
    st.markdown("### Model Drift Uyarisi")
    st.markdown("""
    <div class='hero-card'>
    <b>Uretim Ortami Icin Onerilen Izleme:</b>
    <ul style='margin-top:8px'>
    <li>Gunluk/haftalik tahmin dagilimi takibi (confidence histogram)</li>
    <li>Input feature dagilimlarinin egitim verisiyle karsilastirilmasi</li>
    <li>Gercek churn orani ile model tahmin oraninin karsilastirilmasi</li>
    <li>6-12 ayda bir model yeniden egitimi degerlendirilmesi</li>
    <li>Dusuk guvenli tahmin sayisi artarsa alert tetiklenmesi</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SEKME 6: YARDIM
# ---------------------------------------------------------------------------
def tab_help():
    st.markdown("## Yardim ve Dokumantasyon")

    st.markdown("""
    <div class='hero-card'>
    <h3>Uygulama Nasil Kullanilir?</h3>

    <b>1. Tekil Tahmin:</b> Bir musterinin bilgilerini forme girin ve 'Churn Tahmini Yap' butonuna basin.
    Sonuc olarak churn olasiligi, risk seviyesi ve eylem onerisi goreceksiniz.<br><br>

    <b>2. Toplu Tahmin:</b> CSV dosyasi yukleyin. Dosyaniz beklenen sutunlari icermeli.
    Tahmin sonuclari tablo olarak gosterilir ve CSV olarak indirilebilir.<br><br>

    <b>3. Model Performansi:</b> Tum model karsilastirma grafikleri, confusion matrix ve ROC egrisi.<br><br>

    <b>4. Model Bilgisi:</b> Final model secim gerekceleri, pipeline ozellikleri ve sinirliliklar.<br><br>

    <b>5. Monitoring:</b> Tahmin gecmisi logu, confidence dagilimi ve drift izleme onerileri.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Girdi Alanlari Aciklamalari</div>", unsafe_allow_html=True)
    help_data = pd.DataFrame([
        ["CreditScore",      "300–850",       "Musterinin kredi skoru. Dusuk skor risk gostergesi olabilir."],
        ["Geography",        "France/Spain/Germany", "Musterinin kayitli oldugu ulke."],
        ["Gender",           "Male/Female",   "Musterinin cinsiyeti."],
        ["Age",              "18–95",         "Musterinin yasi. Yas churn ile pozitif korelasyonludur."],
        ["Tenure",           "0–10 yil",      "Bankada gecirilen yil sayisi. Uzun surede bagli musteri."],
        ["Balance",          "0–300.000",     "Musterinin hesap bakiyesi. Sifir bakiye risk sinyali."],
        ["NumOfProducts",    "1–4",           "Bankayla urun sayisi. 3+ yuksek risk faktoru."],
        ["HasCrCard",        "0/1",           "Kredi karti sahipligi (1=Evet)."],
        ["IsActiveMember",   "0/1",           "Son donemde aktif kullanim (1=Evet). Aktif olmayan risk."],
        ["EstimatedSalary",  "USD/yil",       "Musterinin tahmini yillik geliri."],
    ], columns=["Alan", "Aralik/Deger", "Aciklama"])
    st.dataframe(help_data, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Model Sinirliliklar</div>", unsafe_allow_html=True)
    st.warning(
        "Bu model ikili siniflandirma (churn/churn yok) yapar. "
        "Precision degeri goreceli dusuktur (%38.2), yani bazi 'churn edecek' tahminleri yanlis olabilir. "
        "Yuksek riskli kararlar icin modelin sonucuna ek olarak musteri temsilcisi degerlendirmesi onerilir. "
        "Model her 6-12 ayda bir yeniden egitim gerektirabilir."
    )

    st.markdown("<div class='section-header'>Lisans ve Sorumluluk</div>", unsafe_allow_html=True)
    st.info(
        "Bu uygulama arastirma ve karar destek amaclidir. "
        "Tek basina nihai is karari vermek icin kullanilmamalidir. "
        "Hassas musteri verileri icin gizlilik politikasi uygulanmalidir."
    )


# ---------------------------------------------------------------------------
# ANA AKIS
# ---------------------------------------------------------------------------
def main():
    inject_css()
    render_sidebar()

    # Session state baslangic (Shneiderman Kural 6 & 7)
    for key in ["last_prediction", "batch_result", "prediction_done"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Hero baslik - koyu mavi gradyan, beyaz metin, dekoratif geometrik detay
    st.markdown(
        """
        <div class='hero-header'>
            <div class='hero-title'>Musteri Churn Tahmin Sistemi</div>
            <div class='hero-subtitle'>
                Yapay zeka destekli musteri kayip riski analizi ve erken uyari sistemi
            </div>
            <div style='margin-top:14px'>
                <span class='hero-badge'>Logistic Regression</span>
                <span class='hero-badge'>ROC-AUC 0.7629</span>
                <span class='hero-badge'>Recall 0.6716</span>
                <span class='hero-badge'>Ikili Siniflandirma</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model yukleme
    try:
        model, pipeline = load_assets()
    except Exception as e:
        st.error(
            f"Model veya pipeline yuklenemedi: {e}  \n"
            "Lutfen `models/final_model.pkl` ve `models/preprocessing_pipeline.pkl` "
            "dosyalarinin mevcut oldugunu dogrulayin."
        )
        st.stop()

    # Sekmeler (Shneiderman Kural 8 - Bellek yukunu azaltma)
    tabs = st.tabs([
        "Tekil Tahmin",
        "Toplu Tahmin (CSV)",
        "Model Performansi",
        "Model Bilgisi",
        "Monitoring",
        "Yardim",
    ])

    with tabs[0]:
        tab_single(model, pipeline)
    with tabs[1]:
        tab_batch(model, pipeline)
    with tabs[2]:
        tab_performance()
    with tabs[3]:
        tab_model_info()
    with tabs[4]:
        tab_monitoring()
    with tabs[5]:
        tab_help()


if __name__ == "__main__":
    main()
