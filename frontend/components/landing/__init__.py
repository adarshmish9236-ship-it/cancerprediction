"""
CANCERWISE AI — Modular Landing Page Components
Exposes all standardized sections for the CANCERWISE AI landing page experience.
"""

from __future__ import annotations

from frontend.components.landing.navbar import render_navbar
from frontend.components.landing.hero import render_hero
from frontend.components.landing.intro import render_intro
from frontend.components.landing.capabilities import render_capabilities
from frontend.components.landing.how_it_works import render_how_it_works
from frontend.components.landing.technology import render_technology
from frontend.components.landing.analytics_showcase import render_analytics_showcase
from frontend.components.landing.why_cancerwise import render_why_cancerwise
from frontend.components.landing.responsible_ai import render_responsible_ai
from frontend.components.landing.dataset_section import render_dataset_section
from frontend.components.landing.final_cta import render_final_cta
from frontend.components.landing.footer import render_footer

__all__ = [
    "render_navbar",
    "render_hero",
    "render_intro",
    "render_capabilities",
    "render_how_it_works",
    "render_technology",
    "render_analytics_showcase",
    "render_why_cancerwise",
    "render_responsible_ai",
    "render_dataset_section",
    "render_final_cta",
    "render_footer",
]
