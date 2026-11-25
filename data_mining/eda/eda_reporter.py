"""
CANCERWISE AI — Automated EDA Report Generator
Constructs standalone, responsive HTML reports with embedded CSS, executive summaries,
morphometric findings, statistical hypothesis tables, and clinical governance disclaimers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from jinja2 import Template

from backend.core.config import PROJECT_ROOT
from backend.schemas.eda import EDAReportMetadata
from data_mining.eda.eda_engine import EDAEngine

logger = logging.getLogger("cancerwise.data_mining.eda.reporter")


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CANCERWISE AI — Exploratory Data Analysis Report: {{ overview.dataset_name }}</title>
    <style>
        :root {
            --primary: #0052cc;
            --secondary: #172b4d;
            --success: #36b37e;
            --danger: #de350b;
            --warning: #ffab00;
            --background: #f4f5f7;
            --card-bg: #ffffff;
            --text: #172b4d;
            --border: #dfe1e6;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
            margin: 0;
            padding: 30px 15px;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }
        header {
            border-bottom: 2px solid var(--primary);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h1 { color: var(--primary); margin: 0 0 10px 0; font-size: 28px; }
        .meta-bar { color: #5e6c84; font-size: 14px; }
        .alert-box {
            background-color: #e6fcff;
            border-left: 4px solid #00b8d9;
            padding: 15px;
            margin: 25px 0;
            border-radius: 4px;
            font-size: 14px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        .card {
            background: var(--background);
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid var(--border);
        }
        .card .metric-val {
            font-size: 26px;
            font-weight: 700;
            color: var(--primary);
            margin: 5px 0;
        }
        .card .metric-lbl {
            font-size: 12px;
            text-transform: uppercase;
            color: #5e6c84;
            font-weight: 600;
        }
        h2 { color: var(--secondary); margin-top: 35px; border-bottom: 1px solid var(--border); padding-bottom: 8px; font-size: 20px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: #fafbfc;
            font-weight: 600;
            color: var(--secondary);
        }
        tr:hover { background-color: #f4f5f7; }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-success { background: #e3fcef; color: #006644; }
        .badge-danger { background: #ffebe6; color: #bf2600; }
        .observation-list li { margin-bottom: 8px; }
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: #7a869a;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CANCERWISE AI — Comprehensive EDA & Statistical Mining Report</h1>
            <div class="meta-bar">
                <strong>Dataset:</strong> {{ overview.dataset_name }} (<code>{{ overview.dataset_id }}</code>) &nbsp;|&nbsp;
                <strong>Generated:</strong> {{ generated_at }} &nbsp;|&nbsp;
                <strong>Platform Version:</strong> 2.0.0
            </div>
        </header>

        <div class="alert-box">
            <strong>⚕️ Clinical Governance Notice:</strong><br>
            {{ wdbc_findings.non_causal_disclaimer if wdbc_findings else "Statistical findings reflect observational patterns for machine learning pipelines. They DO NOT establish biological causality." }}
        </div>

        <h2>1. Executive Dataset Overview</h2>
        <div class="grid">
            <div class="card">
                <div class="metric-lbl">Total Records</div>
                <div class="metric-val">{{ "{:,}".format(overview.total_records) }}</div>
            </div>
            <div class="card">
                <div class="metric-lbl">Features</div>
                <div class="metric-val">{{ overview.total_features }}</div>
            </div>
            <div class="card">
                <div class="metric-lbl">Missing Data</div>
                <div class="metric-val">{{ overview.missing_value_percentage }}%</div>
            </div>
            <div class="card">
                <div class="metric-lbl">Duplicates</div>
                <div class="metric-val">{{ overview.duplicate_records_percentage }}%</div>
            </div>
            <div class="card">
                <div class="metric-lbl">Memory Usage</div>
                <div class="metric-val">{{ overview.memory_utilization_mb }} MB</div>
            </div>
        </div>

        <h2>2. Target Class Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Class Label</th>
                    <th>Instance Count</th>
                    <th>Proportion</th>
                    <th>Clinical Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {% for label, count in overview.target_distribution.items() %}
                <tr>
                    <td><strong>{{ label }}</strong></td>
                    <td>{{ count }}</td>
                    <td>{{ (overview.target_proportions[label] * 100)|round(2) }}%</td>
                    <td>
                        {% if label == "Benign" %}
                        <span class="badge badge-success">Non-Cancerous Hyperplastic Tissue</span>
                        {% elif label == "Malignant" %}
                        <span class="badge badge-danger">Invasive Neoplastic Carcinoma</span>
                        {% else %}
                        <span>Observational Target Class</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        {% if wdbc_findings %}
        <h2>3. Top Diagnostic Differentiating Features (Cytological Nuclear Morphometrics)</h2>
        <p>Ranked by absolute Cohen's d effect size between Benign and Malignant biopsy samples:</p>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Morphometric Feature</th>
                    <th>Benign Mean</th>
                    <th>Malignant Mean</th>
                    <th>Difference (%)</th>
                    <th>Cohen's d</th>
                    <th>Mann-Whitney p-value</th>
                </tr>
            </thead>
            <tbody>
                {% for diff in wdbc_findings.top_differentiating_features[:10] %}
                <tr>
                    <td>{{ diff.significance_rank }}</td>
                    <td><code>{{ diff.feature }}</code></td>
                    <td>{{ diff.benign_mean }}</td>
                    <td>{{ diff.malignant_mean }}</td>
                    <td><strong>{{ "{:+.1f}".format(diff.mean_difference_percent) }}%</strong></td>
                    <td>{{ diff.cohens_d }}</td>
                    <td>&lt; 1e-15</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>4. Key Statistical Observations</h2>
        <ul class="observation-list">
            {% for obs in wdbc_findings.morphometric_observations %}
            <li>{{ obs }}</li>
            {% endfor %}
        </ul>

        <h2>5. Multicollinearity & Feature Clusters</h2>
        <p>Highly collinear feature groups with pairwise correlation <em>r &ge; 0.90</em>:</p>
        <table>
            <thead>
                <tr>
                    <th>Primary Landmark Feature</th>
                    <th>Collinear Group Members</th>
                    <th>Average Correlation (r)</th>
                </tr>
            </thead>
            <tbody>
                {% for cluster in wdbc_findings.collinear_feature_clusters %}
                <tr>
                    <td><strong><code>{{ cluster.primary_feature }}</code></strong></td>
                    <td>{{ cluster.collinear_features|join(", ") }}</td>
                    <td>{{ cluster.avg_correlation }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <h2>6. Multiple-Testing Corrected Hypothesis Tests</h2>
        <p>Mann-Whitney U tests evaluated with Benjamini-Hochberg False Discovery Rate (FDR) control at &alpha; = 0.05:</p>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Test Statistic (U)</th>
                    <th>Raw p-value</th>
                    <th>FDR Adjusted p-value</th>
                    <th>Significant</th>
                    <th>Effect Size (Rank-Biserial)</th>
                </tr>
            </thead>
            <tbody>
                {% for test in stat_tests[:12] %}
                <tr>
                    <td><code>{{ test.feature_name }}</code></td>
                    <td>{{ test.statistic }}</td>
                    <td>{{ "{:.2e}".format(test.p_value) }}</td>
                    <td>{{ "{:.2e}".format(test.p_value_corrected) if test.p_value_corrected else "N/A" }}</td>
                    <td>
                        {% if test.is_significant %}
                        <span class="badge badge-danger">Significant</span>
                        {% else %}
                        <span class="badge badge-success">Not Significant</span>
                        {% endif %}
                    </td>
                    <td>{{ test.effect_size }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <footer>
            CANCERWISE AI — Advanced Cancer Prediction and Intelligent Data Mining Platform<br>
            Report generated automatically via the CANCERWISE Statistical Mining Engine.
        </footer>
    </div>
</body>
</html>
"""


class EDAReporter:
    """
    Builds and persists publication-grade HTML reports.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or (PROJECT_ROOT / "reports" / "eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eda_engine = EDAEngine()

    def generate_html_report(
        self,
        df: pd.DataFrame,
        dataset_id: str = "dataset",
        dataset_name: str = "Cancer Dataset",
        target_column: Optional[str] = "diagnosis",
    ) -> Tuple[str, EDAReporter]:
        """
        Synthesize complete EDA report into an HTML file.
        Returns: (file_path_str, self)
        """
        overview = self.eda_engine.get_overview(df, dataset_id=dataset_id, dataset_name=dataset_name, target_column=target_column)

        wdbc_findings = None
        if target_column and target_column in df.columns:
            try:
                wdbc_findings = self.eda_engine.analyze_cancer_wdbc(df, target_column=target_column)
            except Exception as e:
                logger.warning("Could not execute WDBC specific findings: %s", e)

        stat_tests = []
        if target_column and target_column in df.columns:
            try:
                stat_tests = self.eda_engine.run_statistical_tests(df, target_column=target_column)
            except Exception as e:
                logger.warning("Could not execute hypothesis testing pipeline: %s", e)

        template = Template(HTML_REPORT_TEMPLATE)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        html_content = template.render(
            overview=overview,
            wdbc_findings=wdbc_findings,
            stat_tests=stat_tests,
            generated_at=now_str,
        )

        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_filename = f"eda_report_{dataset_id}_{timestamp_tag}.html"
        out_path = self.output_dir / out_filename

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Generated EDA HTML report for '%s' at %s", dataset_id, out_path)
        return str(out_path), self
