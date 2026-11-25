# CANCERWISE AI — Unified Data Dictionary

---

> [!CAUTION]
> **Biomedical & Clinical Classification Notice:**
> The target variable `diagnosis` distinguishes between **Benign** (non-malignant) and **Malignant** (carcinoma) cellular pathology.
> **Diagnostic labels must NEVER be confused with cancer staging.**
> Clinical staging (Stage 0 through Stage IV) requires anatomical TNM classification (tumor size, regional lymph node metastasis, distant organ metastases) which is not captured in cytological FNA fluid measurements.

---

## 1. Primary Benchmark Cohort: Wisconsin Diagnostic Breast Cancer (WDBC)

### 1.1 Target Semantics

| Target Value | Raw Code | Binary Integer | Histopathological Definition | Clinical Triage Guideline |
| :--- | :---: | :---: | :--- | :--- |
| **Benign** | `B` | `0` | Non-malignant tissue proliferation (e.g. fibroadenoma, fibrocystic changes, cysts). Nuclei maintain regular nuclear-to-cytoplasmic ratio, uniform chromatin, and smooth borders. | Routine clinical mammography and observational follow-up. |
| **Malignant** | `M` | `1` | Infiltrating ductal or lobular carcinoma. Nuclei exhibit marked pleomorphism, high volume, membrane notches, coarse chromatin clumping, and frequent mitotic figures. | Immediate surgical / oncological referral for histological core needle biopsy and TNM staging. |

---

### 1.2 Feature Taxonomy & Nuclear Morphometrics

The 30 continuous features are derived from 10 fundamental nuclear characteristics computed for each cell nucleus in the digitized fine needle aspirate (FNA):

1. **Radius:** Mean of distances from center to points on the perimeter ($\mu m$).
2. **Texture:** Standard deviation of gray-scale pixel intensities (variance).
3. **Perimeter:** Total boundary perimeter length of the nuclear contour ($\mu m$).
4. **Area:** Total pixel area enclosed by the nuclear membrane ($\mu m^2$).
5. **Smoothness:** Local variation in radius lengths (measured by difference between radial length and mean of neighbouring radii).
6. **Compactness:** Defined as $\frac{\text{perimeter}^2}{\text{area}} - 1.0$ (dimensionless measure of roundness).
7. **Concavity:** Severity and depth of concave indentations along the nuclear perimeter.
8. **Concave Points:** Count of distinct concave indentations and notches along the perimeter.
9. **Symmetry:** Relative difference between perpendicular chords traversing the nuclear centroid.
10. **Fractal Dimension:** Approximation of boundary microscopic complexity ($D - 1.0$ via "coastline approximation").

Each characteristic is summarized across all nuclei in the aspirate in three ways:
- **Mean (`_mean`)**: Population average across all segmented nuclei in the image.
- **Standard Error (`_se`)**: Statistical variation across the cell population (quantifying anisokaryosis).
- **Worst / Largest (`_worst`)**: Average of the three most severe/largest nuclei in the aspirate.

---

### 1.3 Complete WDBC Attribute Matrix

| # | Attribute Name | Storage Type | Physical Units | Feature Group | Descriptive Statistics (Mean ± Std) | Clinical Interpretation |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `id` | int64 | N/A | Identifier | N/A | Anonymized patient sample identifier |
| 2 | `diagnosis` | string / int64 | Categorical | Target | 357 B / 212 M | Ground-truth pathologist histological confirmation |
| 3 | `radius_mean` | float64 | $\mu m$ | Mean | $14.13 \pm 3.52$ | Average nuclear radius |
| 4 | `texture_mean` | float64 | Grayscale std | Mean | $19.29 \pm 4.30$ | Average chromatin condensation & density |
| 5 | `perimeter_mean` | float64 | $\mu m$ | Mean | $91.97 \pm 24.30$ | Average nuclear circumference |
| 6 | `area_mean` | float64 | $\mu m^2$ | Mean | $654.89 \pm 351.91$ | Average nuclear surface area |
| 7 | `smoothness_mean` | float64 | Ratio | Mean | $0.096 \pm 0.014$ | Average membrane contour smoothness |
| 8 | `compactness_mean` | float64 | Ratio | Mean | $0.104 \pm 0.053$ | Nuclear circularity & folding |
| 9 | `concavity_mean` | float64 | Ratio | Mean | $0.089 \pm 0.080$ | Average depth of contour indentations |
| 10 | `concave_points_mean` | float64 | Count | Mean | $0.049 \pm 0.039$ | Average count of membrane notches |
| 11 | `symmetry_mean` | float64 | Ratio | Mean | $0.181 \pm 0.027$ | Nuclear chord balance & symmetry |
| 12 | `fractal_dimension_mean` | float64 | Ratio | Mean | $0.063 \pm 0.007$ | Average border microscopic irregularity |
| 13 | `radius_se` | float64 | $\mu m$ | SE | $0.405 \pm 0.277$ | Cell-to-cell nuclear size variance (anisokaryosis) |
| 14 | `texture_se` | float64 | Grayscale | SE | $1.217 \pm 0.552$ | Staining heterogeneity across cell population |
| 15 | `perimeter_se` | float64 | $\mu m$ | SE | $2.866 \pm 2.022$ | Perimeter variance across sample |
| 16 | `area_se` | float64 | $\mu m^2$ | SE | $40.34 \pm 45.49$ | Nuclear volume heterogeneity |
| 17 | `smoothness_se` | float64 | Ratio | SE | $0.007 \pm 0.003$ | Cell contour variability |
| 18 | `compactness_se` | float64 | Ratio | SE | $0.025 \pm 0.018$ | Morphological irregularity variance |
| 19 | `concavity_se` | float64 | Ratio | SE | $0.032 \pm 0.030$ | Indentation severity variance |
| 20 | `concave_points_se` | float64 | Count | SE | $0.012 \pm 0.006$ | Notch frequency variance |
| 21 | `symmetry_se` | float64 | Ratio | SE | $0.021 \pm 0.008$ | Asymmetry variance between nuclei |
| 22 | `fractal_dimension_se` | float64 | Ratio | SE | $0.004 \pm 0.003$ | Border complexity variance |
| 23 | `radius_worst` | float64 | $\mu m$ | Worst | $16.27 \pm 4.83$ | Largest nuclei radius (hallmark of aggressive clones) |
| 24 | `texture_worst` | float64 | Grayscale | Worst | $25.68 \pm 6.15$ | Maximum chromatin density & clumpiness |
| 25 | `perimeter_worst` | float64 | $\mu m$ | Worst | $107.26 \pm 33.60$ | Largest nuclear circumference |
| 26 | `area_worst` | float64 | $\mu m^2$ | Worst | $880.58 \pm 569.36$ | Maximum nuclear area (most predictive single feature) |
| 27 | `smoothness_worst` | float64 | Ratio | Worst | $0.132 \pm 0.023$ | Maximum membrane jaggedness |
| 28 | `compactness_worst` | float64 | Ratio | Worst | $0.254 \pm 0.157$ | Maximum border convolution |
| 29 | `concavity_worst` | float64 | Ratio | Worst | $0.272 \pm 0.209$ | Deepest nuclear clefts |
| 30 | `concave_points_worst`| float64 | Count | Worst | $0.115 \pm 0.066$ | Maximum count of membrane clefts |
| 31 | `symmetry_worst` | float64 | Ratio | Worst | $0.290 \pm 0.062$ | Maximum nuclear distortion & polarity loss |
| 32 | `fractal_dimension_worst`| float64 | Ratio | Worst | $0.084 \pm 0.018$ | Highest microscopic contour complexity |

---

## 2. Multi-Cohort Extensible Schemas

### 2.1 TCGA-BRCA Genomic Expression Connector Schema
- `sample_id`: Anonymized TCGA barcode (e.g. `TCGA-A1-A0SD-01A`).
- `gene_symbol`: HUGO Gene Nomenclature Committee (HGNC) symbol (e.g. `ESR1`, `PGR`, `ERBB2`, `MKI67`, `TP53`).
- `log2_fpkm_normalized`: Log2-transformed Fragments Per Kilobase of transcript per Million mapped reads.
- `pam50_subtype`: Intrinsic molecular subtype (`LumA`, `LumB`, `HER2`, `Basal`, `Normal`).

### 2.2 BreakHis Histopathological Image Connector Schema
- `slide_id`: Anonymized biopsy glass slide identifier.
- `magnification`: Optical lens objective (`40X`, `100X`, `200X`, `400X`).
- `histological_type`: Pathological subtype (e.g. `fibroadenoma`, `ductal_carcinoma`).
- `binary_target`: Superordinate category (`Benign` vs `Malignant`).
- `image_path`: Relative filesystem URI to high-resolution PNG specimen image.
