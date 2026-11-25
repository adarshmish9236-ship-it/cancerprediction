"""
CANCERWISE AI — Visualization Service
Interactive Plotly and Matplotlib visualization engine for exploratory data analysis.
Generates publication-quality univariate, bivariate, multivariate, and PCA figures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger("cancerwise.data_mining.eda.visualization")


class VisualizationService:
    """
    Constructs rich interactive Plotly figures and static Matplotlib charts.
    """

    COLOR_MAP = {"Benign": "#2ca02c", "Malignant": "#d62728"}
    TEMPLATE = "plotly_white"

    # -----------------------------------------------------------------------
    # Univariate Visualizations
    # -----------------------------------------------------------------------
    @classmethod
    def create_histogram(
        cls,
        df: pd.DataFrame,
        column: str,
        target_column: Optional[str] = None,
        nbins: int = 35,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Interactive histogram with optional target class overlay."""
        color = target_column if (target_column and target_column in df.columns) else None
        color_discrete_map = cls.COLOR_MAP if color else None

        fig = px.histogram(
            df,
            x=column,
            color=color,
            nbins=nbins,
            marginal="box",
            barmode="overlay",
            opacity=0.7,
            color_discrete_map=color_discrete_map,
            template=cls.TEMPLATE,
            title=title or f"Feature Distribution: {column}",
        )
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_violin_plot(
        cls,
        df: pd.DataFrame,
        column: str,
        target_column: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Violin plot showing probability density and inner quartiles."""
        x = target_column if (target_column and target_column in df.columns) else None
        color = target_column if (target_column and target_column in df.columns) else None
        color_discrete_map = cls.COLOR_MAP if color else None

        fig = px.violin(
            df,
            y=column,
            x=x,
            color=color,
            box=True,
            points="outliers",
            color_discrete_map=color_discrete_map,
            template=cls.TEMPLATE,
            title=title or f"Violin Density: {column}",
        )
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_box_plot(
        cls,
        df: pd.DataFrame,
        column: str,
        target_column: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Box plot with jittered individual data points."""
        x = target_column if (target_column and target_column in df.columns) else None
        color = target_column if (target_column and target_column in df.columns) else None
        color_discrete_map = cls.COLOR_MAP if color else None

        fig = px.box(
            df,
            y=column,
            x=x,
            color=color,
            points="all",
            notched=True,
            color_discrete_map=color_discrete_map,
            template=cls.TEMPLATE,
            title=title or f"Quantile Box Plot: {column}",
        )
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_bar_chart(
        cls,
        df: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Bar chart of frequency distributions for categorical variables."""
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]

        fig = px.bar(
            counts,
            x=column,
            y="count",
            text="count",
            color=column,
            color_discrete_map=cls.COLOR_MAP,
            template=cls.TEMPLATE,
            title=title or f"Class Frequencies: {column}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    # -----------------------------------------------------------------------
    # Bivariate Visualizations
    # -----------------------------------------------------------------------
    @classmethod
    def create_scatter_plot(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        target_column: Optional[str] = None,
        add_trendline: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Bivariate scatter plot with optional linear OLS trendline and class hue."""
        color = target_column if (target_column and target_column in df.columns) else None
        color_discrete_map = cls.COLOR_MAP if color else None
        trendline = "ols" if add_trendline else None

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color,
            trendline=trendline,
            opacity=0.8,
            color_discrete_map=color_discrete_map,
            template=cls.TEMPLATE,
            title=title or f"Bivariate Relationship: {x_col} vs {y_col}",
        )
        fig.update_layout(height=460, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    # -----------------------------------------------------------------------
    # Multivariate Visualizations
    # -----------------------------------------------------------------------
    @classmethod
    def create_correlation_heatmap(
        cls,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        title: str = "Feature Correlation Matrix",
    ) -> go.Figure:
        """Annotated correlation heatmap with diverging color scale."""
        num_cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c != "id"][:25]  # Limit to 25 for legible text

        corr = df[num_cols].corr(method=method).round(2)

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            template=cls.TEMPLATE,
            title=f"{title} ({method.capitalize()})",
        )
        fig.update_layout(height=580, margin=dict(l=60, r=60, t=60, b=60))
        return fig

    @classmethod
    def create_pca_projection(
        cls,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        target_column: Optional[str] = "diagnosis",
        n_components: int = 2,
    ) -> Tuple[go.Figure, List[float]]:
        """
        Principal Component Analysis (PCA) projection in 2D or 3D.
        Returns: (figure, explained_variance_ratio)
        """
        cols = numeric_columns or [c for c in df.select_dtypes(include=np.number).columns if c != "id" and c != target_column]
        subset = df[cols].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset)

        pca = PCA(n_components=min(n_components, len(cols)), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        exp_var = [float(v) for v in pca.explained_variance_ratio_]

        color_col = df.loc[subset.index, target_column] if target_column and target_column in df.columns else None

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
        })
        if color_col is not None:
            pca_df[target_column] = color_col.values

        if n_components == 3 and X_pca.shape[1] >= 3:
            pca_df["PC3"] = X_pca[:, 2]
            fig = px.scatter_3d(
                pca_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color=target_column if color_col is not None else None,
                color_discrete_map=cls.COLOR_MAP,
                template=cls.TEMPLATE,
                title=f"3D PCA Projection (Total Explained Variance: {sum(exp_var)*100:.1f}%)",
                labels={
                    "PC1": f"PC1 ({exp_var[0]*100:.1f}%)",
                    "PC2": f"PC2 ({exp_var[1]*100:.1f}%)",
                    "PC3": f"PC3 ({exp_var[2]*100:.1f}%)",
                },
            )
            fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
        else:
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color=target_column if color_col is not None else None,
                color_discrete_map=cls.COLOR_MAP,
                template=cls.TEMPLATE,
                title=f"2D PCA Projection (Total Explained Variance: {sum(exp_var[:2])*100:.1f}%)",
                labels={
                    "PC1": f"PC1 ({exp_var[0]*100:.1f}%)",
                    "PC2": f"PC2 ({exp_var[1]*100:.1f}%)",
                },
            )
            fig.update_layout(height=480, margin=dict(l=40, r=40, t=50, b=40))

        return fig, exp_var

    @classmethod
    def create_cluster_visualization(
        cls,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        n_clusters: int = 2,
    ) -> go.Figure:
        """K-Means clustering visualized in 2D principal component coordinates."""
        cols = numeric_columns or [c for c in df.select_dtypes(include=np.number).columns if c != "id"]
        subset = df[cols].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": [f"Cluster {l}" for l in labels],
        })

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            template=cls.TEMPLATE,
            title=f"K-Means Exploratory Clusters (k={n_clusters}) in PCA Space",
        )
        fig.update_layout(height=460, margin=dict(l=40, r=40, t=50, b=40))
        return fig
