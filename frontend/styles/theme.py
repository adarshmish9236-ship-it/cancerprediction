"""
CANCERWISE AI — Modern Scientific Theme Engine & CSS Architecture
Injects responsive, accessible, high-contrast healthcare research CSS styles
for both Dark Navy and Clean Light modes, and formats Plotly figures.
"""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st
import plotly.graph_objects as go
from frontend.styles.design_tokens import PALETTE, TYPOGRAPHY, SPACING


def get_theme_css(theme_mode: str = "Dark Navy") -> str:
    """Generates the master CSS stylesheet adhering to the clinical design system."""
    is_dark = "Dark" in theme_mode or "navy" in theme_mode.lower()

    # Dynamic variables based on active theme
    bg_app = PALETTE.BG_DARK if is_dark else PALETTE.BG_LIGHT
    bg_surface = PALETTE.SURFACE_DARK if is_dark else PALETTE.SURFACE_WHITE
    bg_surface_alt = PALETTE.SURFACE_DARK_ALT if is_dark else "#EDF2F7"
    border_color = PALETTE.BORDER_DARK if is_dark else PALETTE.BORDER_LIGHT
    text_primary = PALETTE.TEXT_PRIMARY_DARK if is_dark else PALETTE.TEXT_PRIMARY_LIGHT
    text_secondary = PALETTE.TEXT_SECONDARY_DARK if is_dark else PALETTE.TEXT_SECONDARY_LIGHT
    text_muted = PALETTE.TEXT_MUTED_DARK if is_dark else PALETTE.TEXT_MUTED_LIGHT
    card_shadow = "0 4px 14px rgba(0, 0, 0, 0.35)" if is_dark else "0 2px 10px rgba(20, 43, 69, 0.06)"
    hover_shadow = "0 8px 24px rgba(21, 155, 145, 0.18)" if is_dark else "0 6px 16px rgba(20, 43, 69, 0.12)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --cw-primary-navy: {PALETTE.PRIMARY_NAVY};
        --cw-secondary-navy: {PALETTE.SECONDARY_NAVY};
        --cw-primary-teal: {PALETTE.PRIMARY_TEAL};
        --cw-accent-cyan: {PALETTE.ACCENT_CYAN};
        --cw-bg-app: {bg_app};
        --cw-bg-surface: {bg_surface};
        --cw-bg-surface-alt: {bg_surface_alt};
        --cw-border: {border_color};
        --cw-text-primary: {text_primary};
        --cw-text-secondary: {text_secondary};
        --cw-text-muted: {text_muted};
        --cw-success: {PALETTE.SUCCESS};
        --cw-warning: {PALETTE.WARNING};
        --cw-error: {PALETTE.ERROR};
    }}

    /* Global Body & Streamlit Viewport */
    html, body, [class*="css"], .stApp {{
        font-family: {TYPOGRAPHY.FONT_FAMILY_PRIMARY} !important;
        background-color: var(--cw-bg-app) !important;
        color: var(--cw-text-primary) !important;
    }}

    /* Smooth page scroll & typography anti-aliasing */
    * {{
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        box-sizing: border-box;
    }}

    /* Top Streamlit Header Bar Adjustment */
    header[data-testid="stHeader"] {{
        background-color: var(--cw-bg-app) !important;
        border-bottom: 1px solid var(--cw-border) !important;
    }}

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {{
        background-color: {"#0D1622" if is_dark else "#FFFFFF"} !important;
        border-right: 1px solid var(--cw-border) !important;
        padding-top: 1rem !important;
    }}
    section[data-testid="stSidebar"] div.block-container {{
        padding: 1.25rem 1rem !important;
    }}

    /* Hero Showcase Banner */
    .cw-hero-banner {{
        background: { "linear-gradient(135deg, #142B45 0%, #101B2A 60%, #182638 100%)" if is_dark else "linear-gradient(135deg, #FFFFFF 0%, #F4F7FA 100%)" };
        border: 1px solid var(--cw-border);
        border-radius: 14px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: { "0 10px 30px rgba(0, 0, 0, 0.4)" if is_dark else "0 4px 18px rgba(20, 43, 69, 0.08)" };
    }}
    .cw-hero-banner::after {{
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(21, 155, 145, 0.22) 0%, rgba(67, 191, 196, 0.0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .cw-hero-title {{
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: { "#FFFFFF" if is_dark else "#142B45" };
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }}
    .cw-hero-subtitle {{
        font-size: 0.95rem;
        color: var(--cw-text-secondary);
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 1.25rem;
    }}
    .cw-hero-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }}

    /* Pipeline Step Indicator */
    .cw-pipeline-card {{
        background: var(--cw-bg-surface);
        border: 1px solid var(--cw-border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: {card_shadow};
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .cw-pipeline-card:hover {{
        border-color: var(--cw-primary-teal);
        transform: translateY(-2px);
    }}

    /* Metric Cards */
    .cw-metric-card {{
        background: {f"linear-gradient(145deg, {bg_surface} 0%, {bg_surface_alt} 100%)" if is_dark else "#FFFFFF"};
        border: 1px solid var(--cw-border);
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        box-shadow: {card_shadow};
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .cw-metric-card:hover {{
        border-color: var(--cw-primary-teal);
        transform: translateY(-2px);
        box-shadow: {hover_shadow};
    }}
    .cw-metric-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--cw-primary-teal), var(--cw-accent-cyan));
        opacity: 0.85;
    }}
    .cw-metric-title {{
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--cw-text-muted);
        font-weight: 600;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .cw-metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--cw-text-primary);
        line-height: 1.15;
        font-feature-settings: "tnum";
    }}
    .cw-metric-delta {{
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .delta-positive {{ color: var(--cw-success) !important; }}
    .delta-negative {{ color: var(--cw-error) !important; }}
    .delta-neutral {{ color: var(--cw-text-secondary) !important; }}

    /* Content Panels & Cards */
    .cw-card {{
        background: var(--cw-bg-surface);
        border: 1px solid var(--cw-border);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        box-shadow: {card_shadow};
        margin-bottom: 1.25rem;
    }}

    /* Standard Chart Containers */
    .cw-chart-container {{
        background: var(--cw-bg-surface);
        border: 1px solid var(--cw-border);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: {card_shadow};
    }}
    .cw-chart-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--cw-border);
        padding-bottom: 0.5rem;
    }}
    .cw-chart-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--cw-text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Badges & Pills */
    .cw-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        line-height: 1.2;
    }}
    .cw-badge-primary {{
        background: rgba(20, 43, 69, 0.15);
        color: {"#60a5fa" if is_dark else "#142B45"};
        border: 1px solid rgba(67, 191, 196, 0.4);
    }}
    .cw-badge-teal {{
        background: rgba(21, 155, 145, 0.16);
        color: var(--cw-primary-teal);
        border: 1px solid rgba(21, 155, 145, 0.4);
    }}
    .cw-badge-success {{
        background: rgba(33, 151, 107, 0.16);
        color: var(--cw-success);
        border: 1px solid rgba(33, 151, 107, 0.4);
    }}
    .cw-badge-warning {{
        background: rgba(217, 154, 56, 0.16);
        color: var(--cw-warning);
        border: 1px solid rgba(217, 154, 56, 0.4);
    }}
    .cw-badge-danger {{
        background: rgba(217, 83, 79, 0.16);
        color: var(--cw-error);
        border: 1px solid rgba(217, 83, 79, 0.4);
    }}

    /* Clinical Alert Banners */
    .cw-alert-clinical {{
        background: { "rgba(217, 83, 79, 0.12)" if is_dark else "#FFF5F5" };
        border-left: 4px solid var(--cw-error);
        border-radius: 6px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0 1.25rem 0;
        font-size: 0.86rem;
        line-height: 1.5;
        color: { "#FCA5A5" if is_dark else "#9B1C1C" };
    }}
    .cw-alert-info {{
        background: { "rgba(21, 155, 145, 0.12)" if is_dark else "#F0FDF4" };
        border-left: 4px solid var(--cw-primary-teal);
        border-radius: 6px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0 1.25rem 0;
        font-size: 0.86rem;
        line-height: 1.5;
        color: { "#99F6E4" if is_dark else "#065F46" };
    }}

    /* Tabs Standardization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid var(--cw-border) !important;
        margin-bottom: 1rem !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0 !important;
        padding: 0.55rem 1rem !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: var(--cw-text-secondary) !important;
        border: none !important;
        background-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: var(--cw-primary-teal) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--cw-primary-teal) !important;
        background-color: { "rgba(21, 155, 145, 0.08)" if is_dark else "rgba(21, 155, 145, 0.05)" } !important;
    }}

    /* Primary & Secondary Buttons */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, var(--cw-primary-teal) 0%, #118178 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.45rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(21, 155, 145, 0.3) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, var(--cw-accent-cyan) 0%, var(--cw-primary-teal) 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(67, 191, 196, 0.4) !important;
    }}

    /* Clean Dividers */
    hr {{
        border: 0 !important;
        border-top: 1px solid var(--cw-border) !important;
        margin: 1.25rem 0 !important;
    }}

    /* Code & Preformatted Blocks */
    code, pre {{
        font-family: {TYPOGRAPHY.FONT_FAMILY_MONO} !important;
        border-radius: 6px !important;
    }}
    </style>
    """
    return css


def apply_theme(theme_mode: str = "Dark Navy") -> None:
    """Injects the computed CSS style block into the active Streamlit app session."""
    st.markdown(get_theme_css(theme_mode), unsafe_allow_html=True)


def get_plotly_template(theme_mode: str = "Dark Navy") -> Dict[str, Any]:
    """Generates a synchronized Plotly layout dictionary for consistent clinical charts."""
    is_dark = "Dark" in theme_mode or "navy" in theme_mode.lower()
    text_color = PALETTE.TEXT_SECONDARY_DARK if is_dark else PALETTE.TEXT_SECONDARY_LIGHT
    grid_color = "rgba(32, 61, 92, 0.45)" if is_dark else "rgba(227, 234, 240, 0.85)"

    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, sans-serif",
            "color": text_color,
            "size": 12,
        },
        "colorway": list(PALETTE.CHART_PALETTE),
        "xaxis": {
            "gridcolor": grid_color,
            "linecolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 11, "color": text_color},
        },
        "yaxis": {
            "gridcolor": grid_color,
            "linecolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 11, "color": text_color},
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 35},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": text_color},
        },
    }
