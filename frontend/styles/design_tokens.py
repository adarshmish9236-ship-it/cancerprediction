"""
CANCERWISE AI — Clinical Design Tokens & Theme Specification
Establishes standardized colors, typography, spacing, elevations, and layout rules
for a premium healthcare research and artificial intelligence dashboard aesthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ColorPalette:
    """Standardized clinical healthcare color system."""
    # Primary Palette
    PRIMARY_NAVY: str = "#142B45"
    SECONDARY_NAVY: str = "#203D5C"
    PRIMARY_TEAL: str = "#159B91"
    ACCENT_CYAN: str = "#43BFC4"

    # Surfaces & Backgrounds
    BG_LIGHT: str = "#F4F7FA"
    SURFACE_WHITE: str = "#FFFFFF"
    BG_DARK: str = "#101B2A"
    SURFACE_DARK: str = "#182638"
    SURFACE_DARK_ALT: str = "#1D2F44"

    # Borders & Dividers
    BORDER_LIGHT: str = "#E3EAF0"
    BORDER_DARK: str = "#203D5C"
    BORDER_ACCENT: str = "#159B91"

    # Typography & Content
    TEXT_PRIMARY_LIGHT: str = "#243447"
    TEXT_SECONDARY_LIGHT: str = "#718096"
    TEXT_MUTED_LIGHT: str = "#A0AEC0"

    TEXT_PRIMARY_DARK: str = "#F4F7FA"
    TEXT_SECONDARY_DARK: str = "#94A3B8"
    TEXT_MUTED_DARK: str = "#64748B"

    # Semantic Status & Diagnostics
    SUCCESS: str = "#21976B"
    WARNING: str = "#D99A38"
    ERROR: str = "#D9534F"
    INFO: str = "#159B91"

    # Chart Sequences (Clinical Palette)
    CHART_PALETTE: tuple = (
        "#159B91",  # Teal (Primary)
        "#43BFC4",  # Cyan (Secondary)
        "#203D5C",  # Navy
        "#21976B",  # Green (Benign/Success)
        "#D9534F",  # Coral Red (Malignant/Alert)
        "#D99A38",  # Amber (Warning)
        "#6366F1",  # Indigo
        "#8B5CF6",  # Violet
    )


@dataclass(frozen=True)
class SpacingScale:
    """Consistent 8-point geometric spacing scale."""
    XS: str = "4px"
    SM: str = "8px"
    MD: str = "12px"
    LG: str = "16px"
    XL: str = "24px"
    XXL: str = "32px"
    XXXL: str = "48px"


@dataclass(frozen=True)
class Typography:
    """System typography standards."""
    FONT_FAMILY_PRIMARY: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    FONT_FAMILY_MONO: str = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace"
    
    # Font Sizes
    SIZE_XS: str = "0.75rem"    # 12px
    SIZE_SM: str = "0.825rem"   # 13.2px
    SIZE_BASE: str = "0.925rem" # 14.8px
    SIZE_MD: str = "1.05rem"    # 16.8px
    SIZE_LG: str = "1.25rem"    # 20px
    SIZE_XL: str = "1.5rem"     # 24px
    SIZE_2XL: str = "1.85rem"   # 29.6px
    SIZE_3XL: str = "2.25rem"   # 36px


# Instantiated Singletons
PALETTE = ColorPalette()
SPACING = SpacingScale()
TYPOGRAPHY = Typography()
