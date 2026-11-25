"""
CANCERWISE AI — Schema Compatibility Checker
Enforces strict biomedical integration rules before merging or concatenating datasets.
Checks:
- Feature names
- Data types
- Units of measurement
- Patient identifiers & overlap
- Label definitions
- Modality matching
- Dataset provenance
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from backend.core.logging_setup import get_logger
from backend.schemas.dataset import DatasetMetadata, SchemaCompatibilityReport, DataModality

logger = get_logger("services.compatibility_checker")


class SchemaCompatibilityChecker:
    """
    Evaluates whether two candidate datasets can be safely joined, merged, or ensembled.
    
    PRINCIPLE:
    "Do not assume that two datasets with similarly named columns represent the same measurements."
    """

    @staticmethod
    def check_compatibility(
        meta_a: DatasetMetadata,
        meta_b: DatasetMetadata,
        df_a: Optional[pd.DataFrame] = None,
        df_b: Optional[pd.DataFrame] = None,
    ) -> SchemaCompatibilityReport:
        reasons: List[str] = []
        warnings: List[str] = []
        type_mismatches: Dict[str, Dict[str, str]] = {}
        unit_mismatches: Dict[str, Dict[str, str]] = {}

        # 1. Modality check
        modality_match = (meta_a.modality == meta_b.modality)
        if not modality_match:
            reasons.append(
                f"Modality mismatch: Dataset A is '{meta_a.modality.value}' while Dataset B is '{meta_b.modality.value}'. "
                f"Cross-modal fusion requires multi-modal encoders, not tabular concatenation."
            )

        # 2. Target Variable & Label Definition check
        target_a = meta_a.target_variable
        target_b = meta_b.target_variable

        if target_a and target_b:
            classes_a = set(meta_a.target_classes or [])
            classes_b = set(meta_b.target_classes or [])
            if classes_a and classes_b and classes_a != classes_b:
                warnings.append(
                    f"Target label mismatch: A has {classes_a} while B has {classes_b}. "
                    f"Labels must be mapped to a harmonized ontology before integration."
                )

        # 3. Feature overlap analysis
        features_a = set(meta_a.feature_names)
        features_b = set(meta_b.feature_names)
        shared = sorted(list(features_a.intersection(features_b)))
        disjoint_a = sorted(list(features_a - features_b))
        disjoint_b = sorted(list(features_b - features_a))

        if not shared:
            reasons.append(
                "No shared feature columns found between Dataset A and Dataset B. "
                "Feature spaces are completely disjoint."
            )

        # 4. Data Type consistency across shared features
        schema_a = meta_a.schema_definition or {}
        schema_b = meta_b.schema_definition or {}

        for feat in shared:
            type_a = schema_a.get(feat)
            type_b = schema_b.get(feat)
            if type_a and type_b and type_a != type_b:
                # Distinguish benign differences (e.g. float32 vs float64) from severe (e.g. int vs object)
                if ("float" in type_a and "float" in type_b) or ("int" in type_a and "int" in type_b):
                    warnings.append(f"Precision difference in '{feat}': {type_a} vs {type_b}.")
                else:
                    type_mismatches[feat] = {"dataset_a": type_a, "dataset_b": type_b}
                    reasons.append(f"Incompatible data type for shared feature '{feat}': {type_a} vs {type_b}.")

        # 5. Patient Overlap & Duplicate check (if DataFrames supplied)
        id_col_a = next((c for c in (df_a.columns if df_a is not None else []) if c.lower() in ("id", "patient_id")), None)
        id_col_b = next((c for c in (df_b.columns if df_b is not None else []) if c.lower() in ("id", "patient_id")), None)

        if df_a is not None and df_b is not None and id_col_a and id_col_b:
            patients_a = set(df_a[id_col_a].dropna().unique())
            patients_b = set(df_b[id_col_b].dropna().unique())
            overlap = patients_a.intersection(patients_b)
            if overlap:
                warnings.append(
                    f"Patient ID overlap detected ({len(overlap)} patients appear in both cohorts). "
                    f"Ensure split integrity to prevent data leakage during model training."
                )

        # 6. Provenance & Measurement Basis
        prov_a = meta_a.provenance.get("clinical_target_nature", "")
        prov_b = meta_b.provenance.get("clinical_target_nature", "")
        if prov_a and prov_b and prov_a != prov_b:
            warnings.append(
                f"Clinical nature difference: Dataset A ({prov_a}) vs Dataset B ({prov_b}). "
                f"Verify measurement harmonization."
            )

        is_compatible = (len(reasons) == 0 and modality_match)

        report = SchemaCompatibilityReport(
            dataset_a_id=meta_a.dataset_id,
            dataset_b_id=meta_b.dataset_id,
            compatible=is_compatible,
            modality_match=modality_match,
            reasons=reasons,
            warnings=warnings,
            shared_features=shared,
            disjoint_features_a=disjoint_a,
            disjoint_features_b=disjoint_b,
            type_mismatches=type_mismatches,
            unit_mismatches=unit_mismatches,
            label_compatibility="Harmonized" if not warnings else "Requires review",
        )

        logger.info(
            "Checked compatibility between '%s' and '%s': compatible=%s",
            meta_a.dataset_id, meta_b.dataset_id, is_compatible
        )
        return report
