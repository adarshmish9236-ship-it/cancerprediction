"""
CANCERWISE AI — Modern Scientific & Medical Analytics Theme System
Bridge to frontend/styles/theme.py ensuring backward compatibility across all modules.
"""

from __future__ import annotations

from frontend.styles.theme import (
    apply_theme,
    get_theme_css,
    get_plotly_template,
)
from frontend.styles.design_tokens import (
    PALETTE,
    SPACING,
    TYPOGRAPHY,
)

__all__ = [
    "apply_theme",
    "get_theme_css",
    "get_plotly_template",
    "PALETTE",
    "SPACING",
    "TYPOGRAPHY",
]
