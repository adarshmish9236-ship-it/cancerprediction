"""
CANCERWISE AI — Modern Scientific & Medical Analytics Frontend Portal
Master application entry point with categorized clinical navigation hierarchy.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from backend.core.config import get_config
from backend.core.logging_setup import setup_logging, get_logger
from frontend.styles.theme import apply_theme
from frontend.utils.api_client import APIClient

setup_logging()
logger = get_logger("frontend.app")
config = get_config()

# ---------------------------------------------------------------------------
# Global Application Configuration
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="CANCERWISE AI | Clinical Oncology Mining & XAI",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# Initialize Session Defaults
if "selected_dataset" not in st.session_state:
    st.session_state["selected_dataset"] = "uci_wdbc"
if "sb_theme_choice" not in st.session_state:
    st.session_state["sb_theme_choice"] = "Dark Navy"

# Apply active theme
apply_theme(st.session_state["sb_theme_choice"])

# Verify API status
api = APIClient()
st.session_state["backend_online"] = api.is_backend_online()

# ---------------------------------------------------------------------------
# Hierarchical Categorized Navigation
# ---------------------------------------------------------------------------
pages = {
    "OVERVIEW": [
        st.Page("pages/01_Overview_Dashboard.py", title="Dashboard", icon="📊", default=True),
    ],
    "DATA MANAGEMENT": [
        st.Page("pages/02_Dataset_Explorer.py", title="Dataset Explorer", icon="📁"),
        st.Page("pages/03_Data_Wrangling_Studio.py", title="Data Wrangling Studio", icon="🧹"),
        st.Page("pages/15_Data_Quality_Analysis.py", title="Data Quality Analysis", icon="🛡️"),
    ],
    "DATA ANALYTICS": [
        st.Page("pages/04_EDA_Studio.py", title="Exploratory Data Analysis", icon="📈"),
        st.Page("pages/05_Feature_Engineering.py", title="Feature Engineering", icon="⚙️"),
        st.Page("pages/16_Feature_Selection.py", title="Feature Selection", icon="🎯"),
    ],
    "MACHINE LEARNING": [
        st.Page("pages/06_Supervised_Learning_Lab.py", title="Supervised Learning", icon="🤖"),
        st.Page("pages/07_Advanced_Learning_Lab.py", title="Advanced Learning", icon="🧬"),
        st.Page("pages/08_Large_Scale_Learning.py", title="Large-Scale Learning", icon="⚡"),
        st.Page("pages/12_Model_Comparison.py", title="Model Comparison", icon="⚖️"),
    ],
    "INTELLIGENCE": [
        st.Page("pages/11_Cancer_Prediction_Workspace.py", title="Cancer Prediction", icon="🩺"),
        st.Page("pages/10_Explainable_AI.py", title="Explainable AI", icon="🔬"),
        st.Page("pages/09_Model_Evaluation.py", title="Model Evaluation", icon="📐"),
    ],
    "SYSTEM": [
        st.Page("pages/13_Research_Reports.py", title="Research Reports", icon="📜"),
        st.Page("pages/17_Experiment_History.py", title="Experiment History", icon="⏱️"),
        st.Page("pages/14_System_Monitoring.py", title="System Monitoring", icon="🖥️"),
        st.Page("pages/18_Settings.py", title="Settings", icon="⚙️"),
    ],
}

pg = st.navigation(pages)
pg.run()
