"""
CANCERWISE AI — Research Evaluation & Model Validation Reporter
Generates automated, publication-ready research reports in Markdown and HTML formats.
Includes diagnostic metrics, 95% bootstrap confidence intervals, prevalence sensitivity (Bayes' Theorem),
probability calibration curves, SHAP/LIME explainability summaries, and ethical/regulatory governance notices.

GOVERNANCE DIRECTIVE:
All clinical evaluation reports must clearly distinguish statistical association from biological etiology
and must include explicit warnings regarding the impact of disease prevalence on positive predictive value (PPV).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from backend.core.config import PROJECT_ROOT
from backend.schemas.explainability import (
    HealthcareValidationMetrics,
    PrevalenceImpactItem,
    CalibrationResult,
    SHAPExplanationSummary,
    RobustnessAuditReport,
    ResearchReportMetadata,
)

logger = logging.getLogger("cancerwise.evaluation.reporter")


class ResearchReporter:
    """
    Automated generator of research validation reports in Markdown and styled HTML.
    """

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or (PROJECT_ROOT / "reports" / "evaluation")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_validation_report(
        self,
        model_id: str,
        algorithm: str,
        dataset_name: str,
        clinical_metrics: HealthcareValidationMetrics,
        calibration_result: Optional[CalibrationResult] = None,
        shap_summary: Optional[SHAPExplanationSummary] = None,
        robustness_report: Optional[RobustnessAuditReport] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> ResearchReportMetadata:
        """
        Builds Markdown and HTML validation reports and writes them to the reports directory.
        """
        report_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_id = f"report_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        md_content = self._build_markdown(
            report_id=report_id,
            timestamp=report_timestamp,
            model_id=model_id,
            algorithm=algorithm,
            dataset_name=dataset_name,
            metrics=clinical_metrics,
            calib=calibration_result,
            shap=shap_summary,
            robust=robustness_report,
            hparams=hyperparameters or {},
        )

        html_content = self._build_html(
            report_id=report_id,
            timestamp=report_timestamp,
            model_id=model_id,
            algorithm=algorithm,
            dataset_name=dataset_name,
            metrics=clinical_metrics,
            calib=calibration_result,
            shap=shap_summary,
            robust=robustness_report,
            hparams=hyperparameters or {},
        )

        md_file = self.reports_dir / f"{model_id}_validation_report.md"
        html_file = self.reports_dir / f"{model_id}_validation_report.html"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Generated research evaluation report: %s, %s", md_file, html_file)

        return ResearchReportMetadata(
            report_id=report_id,
            model_id=model_id,
            report_path_html=str(html_file.resolve()),
            report_path_md=str(md_file.resolve()),
        )

    def _build_markdown(
        self,
        report_id: str,
        timestamp: str,
        model_id: str,
        algorithm: str,
        dataset_name: str,
        metrics: HealthcareValidationMetrics,
        calib: Optional[CalibrationResult],
        shap: Optional[SHAPExplanationSummary],
        robust: Optional[RobustnessAuditReport],
        hparams: Dict[str, Any],
    ) -> str:
        ci_str = ""
        for m_name, ci_bounds in metrics.confidence_intervals_95.items():
            ci_str += f"- **{m_name.upper()}**: {ci_bounds.get('lower', 0):.4f} — {ci_bounds.get('upper', 0):.4f}\n"

        prev_rows = ""
        for item in metrics.prevalence_sensitivity_analysis:
            prev_rows += f"| {item.hypothetical_prevalence_percent:.1f}% | {item.positive_predictive_value_ppv * 100:.2f}% | {item.negative_predictive_value_npv * 100:.2f}% | {item.clinical_interpretation} |\n"

        shap_rows = ""
        if shap and shap.global_importances:
            for item in shap.global_importances[:10]:
                corr_dir = "Positive (Risk ↑)" if item.correlation_with_risk > 0 else "Negative (Risk ↓)"
                shap_rows += f"| {item.rank} | `{item.feature_name}` | {item.mean_abs_shap:.4f} | {corr_dir} |\n"
        else:
            shap_rows = "| - | No SHAP attributions computed | - | - |\n"

        robust_summary = "Not evaluated"
        if robust:
            robust_summary = f"**Resilience Rating**: `{robust.resilience_rating}` | **Shift Gap**: `{robust.distribution_shift_gap:.4f}` | **Boundary Sensitivity**: `{robust.adversarial_boundary_sensitivity:.4f}`"

        md = f"""# CANCERWISE AI — Clinical Model Validation & Research Report

**Report Identifier:** `{report_id}`  
**Generated On:** {timestamp}  
**Model Identifier:** `{model_id}`  
**Algorithm:** `{algorithm}`  
**Dataset:** `{dataset_name}`  

---

## 1. Executive Summary & Diagnostic Overview

The objective of this evaluation is to establish rigorous clinical reliability, probabilistic calibration,
interpretability, and perturbation resilience for the `{algorithm}` predictive model on `{dataset_name}`.

| Metric | Empirical Cohort Value | Clinical Relevance |
|:-------|:-----------------------|:-------------------|
| **Sensitivity (Recall)** | **{metrics.sensitivity_recall * 100:.2f}%** | Proportion of true malignant cases correctly identified |
| **Specificity** | **{metrics.specificity * 100:.2f}%** | Proportion of benign cases correctly ruled out |
| **False-Negative Rate (FNR)** | **{metrics.false_negative_rate_fnr * 100:.2f}%** | Critical missed malignancy rate ($1 - \\text{{Sensitivity}}$) |
| **False-Positive Rate (FPR)** | **{metrics.false_positive_rate_fpr * 100:.2f}%** | Unnecessary benign referral rate ($1 - \\text{{Specificity}}$) |
| **Positive Predictive Value (PPV)** | **{metrics.positive_predictive_value_ppv * 100:.2f}%** | Diagnostic precision in this study cohort |
| **Negative Predictive Value (NPV)** | **{metrics.negative_predictive_value_npv * 100:.2f}%** | Reliability of negative (benign) triage |
| **ROC-AUC** | **{metrics.roc_auc:.4f}** | Discrimination capacity across all discrimination thresholds |
| **PR-AUC** | **{metrics.pr_auc:.4f}** | Area under the Precision-Recall curve |
| **Brier Score** | **{metrics.brier_score:.4f}** | Mean squared difference between predicted probability and outcome |

### 95% Bootstrap Confidence Intervals (N=100)
{ci_str if ci_str else "*(No bootstrap intervals computed)*"}

---

## 2. Prevalence Sensitivity & Bayes' Theorem Analysis

> **Clinical Warning:** Predictive values (PPV and NPV) depend directly on population disease prevalence. 
> Even with 95% sensitivity and 95% specificity, low-prevalence screening settings yield low PPV due to Bayes' rule.

| Disease Prevalence | Expected PPV | Expected NPV | Clinical Cohort Setting |
|:-------------------|:-------------|:-------------|:------------------------|
{prev_rows}

---

## 3. Probability Calibration & Reliability

- **Expected Calibration Error (ECE):** `{calib.expected_calibration_error_ece if calib else 'N/A'}`
- **Brier Reliability Score:** `{calib.brier_score if calib else metrics.brier_score:.4f}`
- **Calibration Method:** `{calib.method.value if calib else 'uncalibrated'}`

*Well-calibrated probabilities ensure risk predictions match empirical frequencies, preventing overconfident false diagnoses.*

---

## 4. Model Explainability & Feature Attributions (SHAP)

| Rank | Feature Biomarker | Mean |SHAP| (Impact) | Directional Association |
|:-----|:------------------|:-------------------|:------------------------|
{shap_rows}

> **Governance Directive:** SHAP values represent additive predictive contributions to model log-odds.
> They indicate statistical associations, **NOT** direct biological causation or oncogenic mechanisms.

---

## 5. Computational Robustness & Stress Audit

{robust_summary}

> **Disclaimer:** Robustness tests measure stability under synthetic Gaussian noise and missingness (MCAR).
> They do not guarantee clinical generalizability across hospital demographics.

---

## 6. Regulatory & Ethical Governance Notice

1. **Investigational Use:** This software is designed as an algorithmic decision support prototype.
2. **Clinical Staging:** Diagnostic predictions (benign vs malignant) must **NEVER** be confused with cancer staging (TNM classification).
3. **Autonomous Action:** Machine learning outputs must never trigger clinical intervention without certified histopathological and oncologist review.
"""
        return md

    def _build_html(
        self,
        report_id: str,
        timestamp: str,
        model_id: str,
        algorithm: str,
        dataset_name: str,
        metrics: HealthcareValidationMetrics,
        calib: Optional[CalibrationResult],
        shap: Optional[SHAPExplanationSummary],
        robust: Optional[RobustnessAuditReport],
        hparams: Dict[str, Any],
    ) -> str:
        prev_rows = ""
        for item in metrics.prevalence_sensitivity_analysis:
            prev_rows += f"""
            <tr>
                <td>{item.hypothetical_prevalence_percent:.1f}%</td>
                <td><strong>{item.positive_predictive_value_ppv * 100:.2f}%</strong></td>
                <td><strong>{item.negative_predictive_value_npv * 100:.2f}%</strong></td>
                <td>{item.clinical_interpretation}</td>
            </tr>
            """

        shap_rows = ""
        if shap and shap.global_importances:
            for item in shap.global_importances[:10]:
                badge = "badge-danger" if item.correlation_with_risk > 0 else "badge-success"
                txt = "Increases Risk (Malignant)" if item.correlation_with_risk > 0 else "Decreases Risk (Benign)"
                shap_rows += f"""
                <tr>
                    <td>{item.rank}</td>
                    <td><code>{item.feature_name}</code></td>
                    <td>{item.mean_abs_shap:.4f}</td>
                    <td><span class="{badge}">{txt}</span></td>
                </tr>
                """
        else:
            shap_rows = "<tr><td colspan='4'>No SHAP values recorded.</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CANCERWISE AI — Clinical Model Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; background: #f8fafc; margin: 0; padding: 2rem; }}
        .container {{ max-width: 1000px; margin: auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 2.5rem; }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.75rem; margin-top: 0; }}
        h2 {{ color: #1e3a8a; margin-top: 2rem; border-left: 4px solid #3b82f6; padding-left: 0.75rem; }}
        .meta-box {{ display: flex; flex-wrap: wrap; gap: 1.5rem; background: #f1f5f9; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2rem; font-size: 0.95rem; }}
        .meta-item {{ flex: 1; min-width: 180px; }}
        .meta-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: bold; }}
        .meta-value {{ font-size: 1.1rem; color: #0f172a; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.25rem 0; font-size: 0.95rem; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; color: #475569; }}
        tr:hover {{ background: #f8fafc; }}
        .alert-warning {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 6px; margin: 1.5rem 0; color: #92400e; font-size: 0.95rem; }}
        .alert-danger {{ background: #fef2f2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 6px; margin: 1.5rem 0; color: #991b1b; font-size: 0.95rem; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }}
        .badge-success {{ background: #dcfce7; color: #166534; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }}
        code {{ background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; font-family: monospace; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CANCERWISE AI — Research Validation Report</h1>
        
        <div class="meta-box">
            <div class="meta-item">
                <div class="meta-label">Report ID</div>
                <div class="meta-value"><code>{report_id}</code></div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Model Identifier</div>
                <div class="meta-value"><code>{model_id}</code></div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Algorithm</div>
                <div class="meta-value">{algorithm}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Dataset</div>
                <div class="meta-value">{dataset_name}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Timestamp</div>
                <div class="meta-value" style="font-size:0.95rem;">{timestamp}</div>
            </div>
        </div>

        <h2>1. Diagnostic Validation & Performance Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Clinical Metric</th>
                    <th>Cohort Score</th>
                    <th>Clinical Significance</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Sensitivity (Recall)</strong></td>
                    <td><strong style="color:#0284c7;">{metrics.sensitivity_recall * 100:.2f}%</strong></td>
                    <td>Proportion of true malignant malignancies correctly flagged.</td>
                </tr>
                <tr>
                    <td><strong>Specificity</strong></td>
                    <td><strong style="color:#16a34a;">{metrics.specificity * 100:.2f}%</strong></td>
                    <td>Proportion of benign cases accurately ruled out.</td>
                </tr>
                <tr>
                    <td><strong>False-Negative Rate (FNR)</strong></td>
                    <td><span class="badge-danger">{metrics.false_negative_rate_fnr * 100:.2f}%</span></td>
                    <td>Critical missed malignancy rate (1 - Sensitivity).</td>
                </tr>
                <tr>
                    <td><strong>False-Positive Rate (FPR)</strong></td>
                    <td>{metrics.false_positive_rate_fpr * 100:.2f}%</td>
                    <td>Unnecessary biopsy referral rate (1 - Specificity).</td>
                </tr>
                <tr>
                    <td><strong>Positive Predictive Value (PPV)</strong></td>
                    <td>{metrics.positive_predictive_value_ppv * 100:.2f}%</td>
                    <td>Diagnostic precision in this specific cohort.</td>
                </tr>
                <tr>
                    <td><strong>Negative Predictive Value (NPV)</strong></td>
                    <td>{metrics.negative_predictive_value_npv * 100:.2f}%</td>
                    <td>Reliability of negative (benign) triage prediction.</td>
                </tr>
                <tr>
                    <td><strong>ROC-AUC</strong></td>
                    <td><strong>{metrics.roc_auc:.4f}</strong></td>
                    <td>Overall discriminatory capability across all decision cutoffs.</td>
                </tr>
                <tr>
                    <td><strong>Brier Score</strong></td>
                    <td>{metrics.brier_score:.4f}</td>
                    <td>Probabilistic calibration loss (lower is better).</td>
                </tr>
            </tbody>
        </table>

        <h2>2. Prevalence Sensitivity & Bayes' Theorem Analysis</h2>
        <div class="alert-warning">
            <strong>Clinical Warning:</strong> Predictive values vary dramatically as disease prevalence shifts. In low-prevalence screening (0.5%), PPV is markedly reduced compared to specialized oncology clinics (37.3%), despite identical sensitivity and specificity.
        </div>
        <table>
            <thead>
                <tr>
                    <th>Cohort Prevalence</th>
                    <th>Expected PPV</th>
                    <th>Expected NPV</th>
                    <th>Clinical Context</th>
                </tr>
            </thead>
            <tbody>
                {prev_rows}
            </tbody>
        </table>

        <h2>3. Probability Calibration & Reliability</h2>
        <p><strong>Expected Calibration Error (ECE):</strong> <code>{calib.expected_calibration_error_ece if calib else 'N/A'}</code></p>
        <p><strong>Calibration Method:</strong> <code>{calib.method.value if calib else 'uncalibrated'}</code></p>

        <h2>4. Model Explainability & Feature Importance (SHAP)</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Biomarker Feature</th>
                    <th>Mean |SHAP| Attribution</th>
                    <th>Direction of Association</th>
                </tr>
            </thead>
            <tbody>
                {shap_rows}
            </tbody>
        </table>

        <div class="alert-danger">
            <strong>Scientific Governance Notice:</strong> SHAP and LIME attributions reflect empirical mathematical associations with model log-odds. They do NOT establish biological causality or clinical staging status.
        </div>
    </div>
</body>
</html>
"""
        return html
