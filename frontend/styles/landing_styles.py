"""
CANCERWISE AI — Premium Landing Page Design Tokens & CSS Architecture
Implements high-fidelity healthcare SaaS aesthetics, glassmorphic cards,
clinical glowing highlights, interactive animations, and responsive grids.
Fully synchronized with frontend.styles.design_tokens and theme.py.
"""

from __future__ import annotations

from frontend.styles.design_tokens import PALETTE, TYPOGRAPHY, SPACING


def get_landing_css(is_dark: bool = True) -> str:
    """Generates custom stylesheet for the CANCERWISE AI landing page."""
    bg_color = PALETTE.BG_DARK if is_dark else PALETTE.BG_LIGHT
    surface_color = PALETTE.SURFACE_DARK if is_dark else PALETTE.SURFACE_WHITE
    surface_alt = PALETTE.SURFACE_DARK_ALT if is_dark else "#EDF3F7"
    border_color = PALETTE.BORDER_DARK if is_dark else PALETTE.BORDER_LIGHT
    text_primary = PALETTE.TEXT_PRIMARY_DARK if is_dark else PALETTE.TEXT_PRIMARY_LIGHT
    text_secondary = PALETTE.TEXT_SECONDARY_DARK if is_dark else PALETTE.TEXT_SECONDARY_LIGHT
    text_muted = PALETTE.TEXT_MUTED_DARK if is_dark else PALETTE.TEXT_MUTED_LIGHT
    card_shadow = "0 8px 32px rgba(0, 0, 0, 0.35)" if is_dark else "0 4px 20px rgba(16, 42, 67, 0.08)"
    hover_shadow = "0 14px 40px rgba(21, 155, 145, 0.25)" if is_dark else "0 10px 30px rgba(16, 42, 67, 0.14)"
    nav_bg = "rgba(16, 27, 42, 0.95)" if is_dark else "rgba(255, 255, 255, 0.96)"

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    --lp-primary-navy: {PALETTE.PRIMARY_NAVY};
    --lp-secondary-navy: {PALETTE.SECONDARY_NAVY};
    --lp-healthcare-teal: {PALETTE.PRIMARY_TEAL};
    --lp-ai-cyan: {PALETTE.ACCENT_CYAN};
    --lp-soft-bg: {bg_color};
    --lp-surface: {surface_color};
    --lp-surface-alt: {surface_alt};
    --lp-border: {border_color};
    --lp-text-primary: {text_primary};
    --lp-text-secondary: {text_secondary};
    --lp-text-muted: {text_muted};
    --lp-success: {PALETTE.SUCCESS};
    --lp-warning: {PALETTE.WARNING};
    --lp-error: {PALETTE.ERROR};
}}

html {{
    scroll-behavior: smooth;
}}

/* Universal Streamlit Button Styling for Landing Page */
.stButton > button {{
    background: rgba(21, 155, 145, 0.14) !important;
    color: {PALETTE.ACCENT_CYAN} !important;
    border: 1px solid rgba(67, 191, 196, 0.45) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}}

.stButton > button:hover {{
    background: rgba(21, 155, 145, 0.32) !important;
    color: #FFFFFF !important;
    border-color: {PALETTE.ACCENT_CYAN} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(67, 191, 196, 0.3) !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {PALETTE.PRIMARY_TEAL} 0%, {PALETTE.SECONDARY_NAVY} 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(67, 191, 196, 0.6) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(21, 155, 145, 0.4) !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, {PALETTE.ACCENT_CYAN} 0%, {PALETTE.PRIMARY_TEAL} 100%) !important;
    color: #FFFFFF !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 8px 24px rgba(67, 191, 196, 0.55) !important;
}}

/* Top Navigation Bar */
.lp-header-bar {{
    background: {nav_bg};
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--lp-border);
    border-radius: 12px;
    padding: 0.85rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: { "0 4px 20px rgba(0, 0, 0, 0.25)" if is_dark else "0 2px 10px rgba(20, 43, 69, 0.05)" };
}}

.lp-nav-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.lp-nav-logo-icon {{
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: linear-gradient(135deg, {PALETTE.PRIMARY_TEAL} 0%, {PALETTE.PRIMARY_NAVY} 100%);
    border: 1px solid rgba(67, 191, 196, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.45rem;
    box-shadow: 0 4px 14px rgba(21, 155, 145, 0.35);
}}

.lp-nav-brand-title {{
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: var(--lp-text-primary);
    line-height: 1.15;
}}

.lp-nav-brand-tag {{
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--lp-healthcare-teal);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}

.lp-nav-badge {{
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    background: rgba(21, 155, 145, 0.12);
    border: 1px solid rgba(21, 155, 145, 0.35);
    color: var(--lp-ai-cyan);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

/* Section Utilities */
.lp-section {{
    padding: 2.5rem 0 2rem 0;
    position: relative;
}}

.lp-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    background: rgba(21, 155, 145, 0.15);
    border: 1px solid rgba(21, 155, 145, 0.4);
    color: var(--lp-ai-cyan);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
}}

.lp-heading {{
    font-family: 'Inter', sans-serif;
    font-size: 2.15rem;
    font-weight: 800;
    color: var(--lp-text-primary);
    letter-spacing: -0.03em;
    line-height: 1.25;
    margin-bottom: 0.75rem;
}}

.lp-subheading {{
    font-size: 1.02rem;
    color: var(--lp-text-secondary);
    line-height: 1.6;
    max-width: 780px;
    margin-bottom: 1.75rem;
}}

.lp-subheading-center {{
    margin-left: auto;
    margin-right: auto;
    text-align: center;
}}

/* Hero Section */
.lp-hero-title {{
    font-family: 'Inter', sans-serif;
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.18;
    color: var(--lp-text-primary);
    margin-bottom: 1.1rem;
}}

.lp-hero-title-accent {{
    background: linear-gradient(135deg, {PALETTE.ACCENT_CYAN} 0%, {PALETTE.PRIMARY_TEAL} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.lp-hero-desc {{
    font-size: 1.08rem;
    line-height: 1.65;
    color: var(--lp-text-secondary);
    margin-bottom: 1.75rem;
}}

.lp-hero-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.25rem;
    margin-bottom: 1.25rem;
}}

.lp-hero-disclaimer {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.8rem;
    color: var(--lp-text-muted);
    background: { "rgba(217, 154, 56, 0.1)" if is_dark else "#FFFBF0" };
    border: 1px solid rgba(217, 154, 56, 0.3);
    border-radius: 8px;
    padding: 0.65rem 1rem;
    line-height: 1.45;
    margin-top: 1.25rem;
}}

/* Diagnostic Widget Showcase (Hero Right) */
.lp-diag-card {{
    background: { "linear-gradient(145deg, #182638 0%, #101B2A 100%)" if is_dark else "linear-gradient(145deg, #FFFFFF 0%, #F4F7FA 100%)" };
    border: 1px solid var(--lp-border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: {card_shadow};
    position: relative;
    overflow: hidden;
}}

.lp-diag-card::after {{
    content: "";
    position: absolute;
    top: -50px;
    right: -50px;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(21, 155, 145, 0.2) 0%, rgba(67, 191, 196, 0.0) 70%);
    border-radius: 50%;
    pointer-events: none;
}}

.lp-diag-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--lp-border);
}}

/* Intro & 3 Pillars */
.lp-pillars-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-top: 1.5rem;
}}

.lp-pillar-card {{
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: 14px;
    padding: 1.75rem;
    box-shadow: {card_shadow};
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}}

.lp-pillar-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--lp-healthcare-teal), var(--lp-ai-cyan));
}}

.lp-pillar-card:hover {{
    transform: translateY(-4px);
    border-color: var(--lp-ai-cyan);
    box-shadow: {hover_shadow};
}}

.lp-pillar-icon {{
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(21, 155, 145, 0.15);
    border: 1px solid rgba(21, 155, 145, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1.25rem;
}}

.lp-pillar-title {{
    font-size: 1.22rem;
    font-weight: 700;
    color: var(--lp-text-primary);
    margin-bottom: 0.65rem;
}}

.lp-pillar-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}

.lp-pillar-list li {{
    font-size: 0.88rem;
    color: var(--lp-text-secondary);
    padding: 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.lp-pillar-list li::before {{
    content: "✦";
    color: var(--lp-healthcare-teal);
    font-size: 0.75rem;
}}

/* Capabilities 8-Grid */
.lp-cap-card {{
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: 14px;
    padding: 1.35rem;
    box-shadow: {card_shadow};
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    min-height: 275px;
    margin-bottom: 0.65rem;
}}

.lp-cap-card:hover {{
    transform: translateY(-3px);
    border-color: var(--lp-ai-cyan);
    box-shadow: {hover_shadow};
}}

.lp-cap-icon-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}}

.lp-cap-icon {{
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: rgba(67, 191, 196, 0.14);
    border: 1px solid rgba(67, 191, 196, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
}}

.lp-cap-badge {{
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--lp-healthcare-teal);
    background: rgba(21, 155, 145, 0.12);
    border: 1px solid rgba(21, 155, 145, 0.25);
    padding: 0.22rem 0.55rem;
    border-radius: 6px;
}}

.lp-cap-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--lp-text-primary);
    margin-bottom: 0.45rem;
}}

.lp-cap-desc {{
    font-size: 0.83rem;
    color: var(--lp-text-secondary);
    line-height: 1.48;
    margin-bottom: 0.85rem;
    flex-grow: 1;
}}

.lp-cap-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 0.5rem;
}}

.lp-cap-tag {{
    font-size: 0.7rem;
    padding: 0.18rem 0.5rem;
    background: {surface_alt};
    color: var(--lp-text-muted);
    border-radius: 4px;
    border: 1px solid var(--lp-border);
}}

/* Timeline Process Section (4 Phases) */
.lp-timeline-track {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.25rem;
    margin-top: 1.5rem;
}}

.lp-timeline-node {{
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: 12px;
    padding: 1.35rem;
    position: relative;
    box-shadow: {card_shadow};
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

.lp-timeline-node:hover {{
    transform: translateY(-3px);
    border-color: var(--lp-healthcare-teal);
}}

.lp-step-number {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--lp-ai-cyan);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.lp-step-title {{
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--lp-text-primary);
    margin-bottom: 0.45rem;
}}

.lp-step-desc {{
    font-size: 0.83rem;
    color: var(--lp-text-secondary);
    line-height: 1.45;
}}

/* Technology Section */
.lp-tech-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 1.5rem;
}}

.lp-tech-card {{
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: 14px;
    padding: 1.65rem;
    box-shadow: {card_shadow};
}}

.lp-tech-card-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--lp-text-primary);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.lp-tech-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.lp-tech-badge {{
    background: {surface_alt};
    border: 1px solid var(--lp-border);
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--lp-text-primary);
    display: flex;
    align-items: center;
    gap: 6px;
    transition: border-color 0.2s ease, background 0.2s ease;
}}

.lp-tech-badge:hover {{
    border-color: var(--lp-ai-cyan);
    background: rgba(67, 191, 196, 0.1);
}}

/* Values Grid */
.lp-values-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
}}

.lp-value-card {{
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: 12px;
    padding: 1.45rem;
    box-shadow: {card_shadow};
}}

.lp-value-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--lp-text-primary);
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.lp-value-desc {{
    font-size: 0.85rem;
    color: var(--lp-text-secondary);
    line-height: 1.5;
}}

.lp-disclaimer-box {{
    background: { "rgba(217, 83, 79, 0.12)" if is_dark else "#FFF5F5" };
    border-left: 4px solid var(--lp-error);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-top: 1.75rem;
    color: { "#FCA5A5" if is_dark else "#9B1C1C" };
    font-size: 0.88rem;
    line-height: 1.6;
}}

/* Dataset Section Cards */
.lp-dataset-card {{
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: 14px;
    padding: 1.65rem;
    box-shadow: {card_shadow};
    margin-bottom: 1.25rem;
}}

.lp-dataset-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid var(--lp-border);
    padding-bottom: 0.85rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;
}}

.lp-meta-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: {surface_alt};
    border: 1px solid var(--lp-border);
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    color: var(--lp-text-primary);
}}

/* Final CTA Banner */
.lp-final-cta {{
    background: linear-gradient(135deg, {PALETTE.PRIMARY_NAVY} 0%, {PALETTE.PRIMARY_TEAL} 100%);
    border: 1px solid rgba(67, 191, 196, 0.4);
    border-radius: 18px;
    padding: 3rem 2.25rem;
    text-align: center;
    color: #FFFFFF;
    box-shadow: 0 16px 48px rgba(21, 155, 145, 0.3);
    position: relative;
    overflow: hidden;
}}

.lp-final-cta-title {{
    font-size: 2.25rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    margin-bottom: 0.85rem;
    line-height: 1.2;
}}

.lp-final-cta-desc {{
    font-size: 1.05rem;
    color: rgba(255, 255, 255, 0.9);
    max-width: 680px;
    margin: 0 auto 1.75rem auto;
    line-height: 1.6;
}}

/* Footer */
.lp-footer {{
    background: { "#0D1622" if is_dark else "#FFFFFF" };
    border-top: 1px solid var(--lp-border);
    padding: 3rem 0 2rem 0;
    margin-top: 3.5rem;
}}

.lp-footer-grid {{
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1.2fr;
    gap: 2rem;
    margin-bottom: 2rem;
}}

.lp-footer-col-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--lp-text-primary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.85rem;
}}

.lp-footer-links {{
    list-style: none;
    padding: 0;
    margin: 0;
}}

.lp-footer-links li {{
    margin-bottom: 0.55rem;
}}

.lp-footer-link {{
    text-decoration: none;
    color: var(--lp-text-secondary);
    font-size: 0.84rem;
    transition: color 0.2s ease;
}}

.lp-footer-link:hover {{
    color: var(--lp-ai-cyan);
}}

.lp-footer-bottom {{
    border-top: 1px solid var(--lp-border);
    padding-top: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.78rem;
    color: var(--lp-text-muted);
}}

/* Responsive Media Queries */
@media (max-width: 992px) {{
    .lp-hero-title {{
        font-size: 2.15rem;
    }}
    .lp-pillars-grid {{
        grid-template-columns: 1fr;
    }}
    .lp-timeline-track {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .lp-values-grid {{
        grid-template-columns: 1fr;
    }}
    .lp-footer-grid {{
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
    }}
}}

@media (max-width: 600px) {{
    .lp-hero-title {{
        font-size: 1.85rem;
    }}
    .lp-timeline-track {{
        grid-template-columns: 1fr;
    }}
    .lp-tech-grid {{
        grid-template-columns: 1fr;
    }}
    .lp-footer-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>"""
