# CANCERWISE AI — Data Dictionary
## Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

**Source:** [UCI Machine Learning Repository #17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
**Origin:** University of Wisconsin Hospitals, Madison (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian, 1995)  
**Modality:** Tabular Continuous Cytological Morphometrics (FNA Biopsy Images)  
**Total Instances:** 569  
**Total Attributes:** 32 (1 ID, 1 Diagnosis Target, 30 Continuous Nuclear Morphometrics)  

---

> [!CAUTION]
> **Biomedical & Clinical Classification Notice:**
> The target variable `diagnosis` distinguishes between **Benign** (non-malignant) and **Malignant** (carcinoma) cellular pathology.
> **Diagnostic labels must NEVER be confused with cancer staging.**
> Clinical staging (Stage 0 through Stage IV) requires anatomical TNM classification (tumor size, regional lymph node metastasis, distant organ metastases) which is not captured in cytological FNA fluid measurements.

---

## Target Variable

| Value | Clinical Code | Description | Clinical Action |
|-------|---------------|-------------|-----------------|
| **Benign** | `B` | Non-malignant tissue proliferation (e.g. fibroadenoma, fibrocystic changes). Nuclei maintain normal size, regular membranes, and uniform chromatin. | Routine clinical follow-up or observation |
| **Malignant** | `M` | Infiltrating carcinomatous malignancy. Nuclei exhibit marked pleomorphism, high area, membrane notches, and irregular hyperchromatic texture. | Immediate surgical / oncological referral for staging (TNM) and biopsy histology |

---

## Feature Taxonomy

The 30 continuous features are derived from 10 fundamental nuclear characteristics computed for each cell nucleus in the digitized fine needle aspirate (FNA):

1. **Radius:** Mean of distances from center to points on the perimeter.
2. **Texture:** Standard deviation of gray-scale values.
3. **Perimeter:** Total boundary perimeter of the cell nucleus.
4. **Area:** Total pixel area within the nuclear contour.
5. **Smoothness:** Local variation in radius lengths.
6. **Compactness:** $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.
7. **Concavity:** Severity of concave indentations along the nuclear contour.
8. **Concave Points:** Number of distinct concave notches along the perimeter.
9. **Symmetry:** Relative difference between perpendicular chords bisecting the nucleus.
10. **Fractal Dimension:** Approximation of boundary complexity ($D - 1.0$).

Each characteristic is summarized across all nuclei in the aspirate in three ways:
- **Mean:** Average across all segmented nuclei in the image (`_mean`)
- **Standard Error:** Statistical variation across the cell population (`_se`)
- **Worst / Largest:** Average of the three most severe/largest nuclei (`_worst`)

---

## Complete Attribute Matrix

| # | Attribute Name | Type | Units | Feature Category | Clinical Interpretation |
|---|----------------|------|-------|------------------|-------------------------|
| 1 | `id` | int64 | N/A | Identifier | Anonymized patient biopsy sample number |
| 2 | `diagnosis` | string | Categorical | **Target** | Pathologist-confirmed diagnostic classification |
| 3 | `radius_mean` | float64 | $\mu m$ | Mean | Average nuclear radius |
| 4 | `texture_mean` | float64 | Grayscale variance | Mean | Average chromatin condensation & density |
| 5 | `perimeter_mean` | float64 | $\mu m$ | Mean | Average nuclear circumference |
| 6 | `area_mean` | float64 | $\mu m^2$ | Mean | Average nuclear surface area |
| 7 | `smoothness_mean` | float64 | Local variance | Mean | Average membrane contour smoothness |
| 8 | `compactness_mean` | float64 | Ratio | Mean | Nuclear circularity & folding |
| 9 | `concavity_mean` | float64 | Ratio | Mean | Average depth of contour indentations |
| 10 | `concave_points_mean` | float64 | Count | Mean | Average count of membrane notches |
| 11 | `symmetry_mean` | float64 | Ratio | Mean | Nuclear chord balance & symmetry |
| 12 | `fractal_dimension_mean` | float64 | Ratio | Mean | Average border microscopic irregularity |
| 13 | `radius_se` | float64 | $\mu m$ | SE | Cell-to-cell nuclear size variance (anisokaryosis) |
| 14 | `texture_se` | float64 | Grayscale | SE | Staining heterogeneity across cell population |
| 15 | `perimeter_se` | float64 | $\mu m$ | SE | Perimeter variance across sample |
| 16 | `area_se` | float64 | $\mu m^2$ | SE | Nuclear volume heterogeneity |
| 17 | `smoothness_se` | float64 | Ratio | SE | Cell contour variability |
| 18 | `compactness_se` | float64 | Ratio | SE | Morphological irregularity variance |
| 19 | `concavity_se` | float64 | Ratio | SE | Indentation severity variance |
| 20 | `concave_points_se` | float64 | Count | SE | Notch frequency variance |
| 21 | `symmetry_se` | float64 | Ratio | SE | Asymmetry variance between nuclei |
| 22 | `fractal_dimension_se` | float64 | Ratio | SE | Border complexity variance |
| 23 | `radius_worst` | float64 | $\mu m$ | Worst | Largest nuclei radius (hallmark of aggressive clones) |
| 24 | `texture_worst` | float64 | Grayscale | Worst | Maximum chromatin density & clumpiness |
| 25 | `perimeter_worst` | float64 | $\mu m$ | Worst | Largest nuclear circumference |
| 26 | `area_worst` | float64 | $\mu m^2$ | Worst | Maximum nuclear area (most predictive single feature) |
| 27 | `smoothness_worst` | float64 | Ratio | Worst | Maximum membrane jaggedness |
| 28 | `compactness_worst` | float64 | Ratio | Worst | Maximum border convolution |
| 29 | `concavity_worst` | float64 | Ratio | Worst | Deepest nuclear clefts |
| 30 | `concave_points_worst`| float64 | Count | Worst | Maximum count of membrane clefts |
| 31 | `symmetry_worst` | float64 | Ratio | Worst | Maximum nuclear distortion & polarity loss |
| 32 | `fractal_dimension_worst`| float64 | Ratio | Worst | Highest microscopic contour complexity |
