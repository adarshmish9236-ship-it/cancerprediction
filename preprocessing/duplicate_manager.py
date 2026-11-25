"""
CANCERWISE AI — Duplicate Manager
Comprehensive duplicate detection, identifier auditing, and similarity investigation.

CRITICAL RULE: Records with identical features must not be automatically deleted
if they may represent distinct clinical patients or serial biopsy encounters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from backend.schemas.preprocessing import DuplicateAuditReport

logger = logging.getLogger("cancerwise.preprocessing.duplicates")


class DuplicateManager:
    """
    Manages exact duplicates, identifier conflicts, and near-identical records.
    Enforces clinical preservation warnings before record removal.
    """

    def __init__(
        self,
        id_columns: Optional[List[str]] = None,
        similarity_threshold: float = 0.99,
    ) -> None:
        self.id_columns = id_columns or ["id", "patient_id", "sample_id", "case_id"]
        self.similarity_threshold = similarity_threshold

    def find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the primary patient or sample identifier column."""
        for candidate in self.id_columns:
            if candidate in df.columns:
                return candidate
        return None

    def audit_duplicates(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        check_similarity: bool = True,
    ) -> DuplicateAuditReport:
        """
        Generate a comprehensive duplicate audit report across exact records,
        patient identifiers, and feature-space similarity.
        """
        id_col = id_column or self.find_id_column(df)
        total_records = len(df)

        if total_records == 0:
            return DuplicateAuditReport(
                total_records=0,
                exact_duplicates_count=0,
                duplicate_identifier_count=0,
                identifier_column=id_col,
            )

        # 1. Exact Duplicate Records
        exact_dup_mask = df.duplicated(keep=False)
        exact_dup_count = int(exact_dup_mask.sum())
        exact_dup_indices = df.index[exact_dup_mask].tolist()

        # 2. Duplicate Identifiers
        duplicate_id_count = 0
        duplicate_id_groups: Dict[str, List[int]] = {}

        if id_col and id_col in df.columns:
            id_dup_mask = df[id_col].duplicated(keep=False)
            duplicate_id_count = int(id_dup_mask.sum())
            if duplicate_id_count > 0:
                for val, group in df[id_dup_mask].groupby(id_col):
                    duplicate_id_groups[str(val)] = group.index.tolist()

        # 3. Similar Record Investigation (on standardized numerical features)
        similar_count = 0
        if check_similarity and total_records > 1:
            num_df = df.select_dtypes(include=np.number)
            if id_col and id_col in num_df.columns:
                num_df = num_df.drop(columns=[id_col])

            if num_df.shape[1] >= 2 and total_records <= 2000:
                # Fill missing temporarily for distance calculation
                filled = num_df.fillna(num_df.median())
                scaler = StandardScaler()
                scaled = scaler.fit_transform(filled)
                sim_matrix = cosine_similarity(scaled)
                # Zero out diagonal
                np.fill_diagonal(sim_matrix, 0.0)
                # Count pairs exceeding similarity threshold
                pairs = np.argwhere(sim_matrix >= self.similarity_threshold)
                similar_count = len(pairs) // 2

        logger.info(
            "Duplicate audit complete | exact=%d | duplicate_ids=%d | similar_pairs=%d",
            exact_dup_count, duplicate_id_count, similar_count,
        )

        return DuplicateAuditReport(
            total_records=total_records,
            exact_duplicates_count=exact_dup_count,
            duplicate_identifier_count=duplicate_id_count,
            identifier_column=id_col,
            duplicate_record_indices=exact_dup_indices,
            duplicate_identifier_groups=duplicate_id_groups,
            similar_records_count=similar_count,
        )

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: Union[str, bool] = "first",
        id_column: Optional[str] = None,
        confirm_clinical_purging: bool = False,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Safely remove duplicate records.

        If distinct patient identifiers exist, requires explicit confirmation
        (confirm_clinical_purging=True) to avoid deleting separate patients with coincidentally
        identical measurements.
        """
        id_col = id_column or self.find_id_column(df)
        before_count = len(df)

        if not confirm_clinical_purging:
            # If subset does not include id_column and id_column exists, warn/block
            if id_col and id_col in df.columns:
                unique_ids = df[id_col].nunique()
                if unique_ids == before_count and subset is not None and id_col not in subset:
                    raise PermissionError(
                        f"Refusing to remove duplicates: Column '{id_col}' confirms all {before_count} records "
                        "have unique patient identifiers. Records represent distinct biological patients. "
                        "Set confirm_clinical_purging=True if deduplication across patients is clinically intended."
                    )

        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore[arg-type]
        removed_count = before_count - len(df_cleaned)

        logger.info("Removed %d duplicate records (remaining: %d).", removed_count, len(df_cleaned))
        return df_cleaned, removed_count
