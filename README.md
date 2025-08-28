# NASA TOPS-T ScienceCore: AI/ML in Space Biology — Notebook Suite

> Consolidated README generated from the notebooks in this folder.

## Overview
This notebook suite was completed as part of the **NASA Transform to Open Science (TOPS-T) – ScienceCore: AI/ML in Space Biology Training**.  
It walks from data fundamentals → visualization/EDA → tabular ML (classification & regression) → image data → clustering/unsupervised learning → explainable AI (XAI), with emphasis on open-science practices and NASA space-biology datasets (OSDR / GeneLab).

---

## Notebook Index (what each file covers)

- **Copy_of_intro_to_data.ipynb**  
  *Intro to data types & metadata; context from space-biology use cases (e.g., SANS).*

- **Copy_of_Data_Visualization.ipynb**  
  *Exploratory plots, distributions, pairplots, correlation heatmaps; communicating results.*

- **Copy of Tabular_Data.ipynb**  
  *Loading/cleaning tabular data; pandas DataFrames; basic feature engineering.*

- **Copy_of_classification.ipynb**  
  *Supervised learning for discrete labels (train/test split, metrics, model comparison).*

- **Copy_of_regression.ipynb**  
  *Supervised learning for continuous targets (error metrics, baselines, regularization).*

- **Copy_of_Image_Data.ipynb**  
  *Bioimaging pipelines: reading, preprocessing, and basic deep-learning scaffolding.*

- **Copy_of_clustering.ipynb**  
  *Unsupervised learning; dimensionality reduction (e.g., PCA), clustering & evaluation.*

- **Copy_of_explainable_ai.ipynb**  
  *XAI for space-bio signals: SHAP/LIME, permutation importance, partial dependence.*

- **Copy_of_bioinformatic_tools.ipynb**  
  *Space-bio tooling overview; dimensionality reduction; ties to downstream ML/XAI.*

> Note: Section headings inside the notebooks include items like **“Mission of the notebook”**, “Read in data and metadata”, “Use PCA to cluster RNA-seq data”, and **“Logistic regression with SHAP for predicting markers using RNA-seq”** (exact phrasing varies per notebook).

---

## Datasets observed in the notebooks

NASA Open Science Data Repository (OSDR) / GeneLab references:
- **OSD-255** (NASA OSDR)
- **OSD-557** / **OSDR-557** (NASA OSDR)
- **OSD-568** (NASA OSDR)
- **OSD-583** (NASA OSDR)
- GeneLab references (e.g., “genelab” in code)
- Programmatic access via **OSDR API** (a URL pattern like `https://osdr.nasa.gov/.../characteristics` appears in data-loading code)

> If you used additional classroom/local CSVs/XLSX files during sessions, add them here with a one-line description.

---

## Methods, Algorithms & Techniques

### Data handling & EDA
- **pandas**, **numpy** for tabular data
- **matplotlib**, **seaborn** for visualization (histograms, box/violin plots, pairplots, correlation heatmaps)

### Preprocessing
- Feature scaling/normalization: **StandardScaler**, **MinMaxScaler**, **RobustScaler**
- **train_test_split**; **Pipeline** / **ColumnTransformer** patterns (where applicable)

### Dimensionality reduction
- **PCA** (principal component analysis)  
  *(t-SNE/UMAP may be added if used in future iterations.)*

### Supervised learning — Classification
- **LogisticRegression**, **SVC**, **KNeighborsClassifier**
- **DecisionTreeClassifier**, **RandomForestClassifier**
- Linear models like **Perceptron** / **SGDClassifier** (as baselines)
- Model selection: **GridSearchCV** / **RandomizedSearchCV** (when present)

**Key metrics:** `accuracy_score`, `f1_score`, ROC-AUC, `confusion_matrix`, `classification_report`

### Supervised learning — Regression
- **LinearRegression**, **Ridge**, **Lasso** (regularization)
- Tree-based/ensemble regressors when included
- **SVR** / **SGDRegressor** (where used)

**Key metrics:** `mean_absolute_error` (MAE), `mean_squared_error` (MSE), `r2_score`

### Unsupervised learning
- **KMeans**, density/hierarchical variants when used
- **silhouette_score** (cluster quality)
- PCA projections for visualization of clusters

### Deep learning (image notebook)
- Framework scaffolding with **PyTorch** and/or **Keras/TensorFlow** (e.g., `nn.Conv2d`, `Sequential`, `Conv2D`, `Dense`)
- Basic image preprocessing (PIL / scikit-image)

### Explainable AI (XAI)
- **SHAP** (e.g., `KernelExplainer`, `TreeExplainer` where appropriate)
- **LIME** (e.g., `LimeTabularExplainer` / `LimeImageExplainer` if present)
- **Permutation importance**
- **Partial dependence** (when included)

---

## Libraries detected across notebooks
- **pandas**, **numpy**, **matplotlib**, **seaborn**
- **scikit-learn** (`sklearn`)
- **shap**, **lime** (XAI)
- **PyTorch** and/or **Keras / TensorFlow** (DL scaffolding)
- **scikit-image** (`skimage`), **PIL**
- Cloud/storage helpers like **s3fs** (where present)
- Jupyter conveniences (`import_ipynb`, etc.)

> Exact imports vary by notebook; this list reflects the common stack observed.

---

## Reproducibility & Open-Science Practices
- Usage of **open datasets** (OSDR / GeneLab IDs cited in code).
- Clear cell ordering and markdown annotations to promote **transparent workflows**.
- Explicit model metrics for **comparability** across runs.
- Programmatic access patterns (e.g., OSDR API) for **repeatable data retrieval**.

---

## How to run locally
1. Create a Python 3.10+ environment and install the dependencies above (`pip install -U pandas numpy matplotlib seaborn scikit-learn shap lime torch torchvision torchaudio tensorflow keras scikit-image s3fs` as needed).
2. Launch Jupyter: `jupyter lab` or `jupyter notebook`.
3. Open a notebook and run top-to-bottom.  
   - For OSDR/GeneLab data, ensure network access and any required API permissions/paths.
   - If a dataset is missing, consult the data-loading cell for links/instructions.

---

## Appendix — Per-notebook highlights (auto-extracted)

- **Copy_of_intro_to_data.ipynb**  
  Headings include items like: *Introduction to data*, space-bio context (e.g., **SANS**).  
  Methods observed: EDA/visualization.

- **Copy_of_Data_Visualization.ipynb**  
  Headings include: *Introduction*, *Histograms/Distributions*, *Correlation Matrix/Heatmap*.  
  Methods observed: EDA/visualization.

- **Copy of Tabular_Data.ipynb**  
  Headings include: *Working with Tabular Data*, *Pandas DataFrame*.  
  Methods observed: Feature scaling/normalization, EDA.

- **Copy_of_classification.ipynb**  
  Headings include: *Mission of the classification notebook*, *Read in data/metadata*.  
  Algorithms/metrics mentioned: `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `accuracy_score`, `confusion_matrix`.  
  Methods observed: Train/Test split, EDA, XAI (SHAP/LIME/permutation importance where used).

- **Copy_of_regression.ipynb**  
  Headings include: *Mission of regression notebook*, *Read in data/metadata*.  
  Algorithms/metrics mentioned: `LinearRegression`, `Ridge`, `MAE`, `MSE`, `r2_score`.  
  Methods observed: Feature scaling, regression evaluation, XAI (SHAP/permutation importance where used).

- **Copy_of_Image_Data.ipynb**  
  Headings include: *Introduction to Bioimaging Data*, *Image Preprocessing*.  
  Methods observed: EDA; image pipelines; DL scaffolding (`torch` / `keras`).

- **Copy_of_clustering.ipynb**  
  Headings include: *Mission of clustering notebook*, *Read in metadata*, *PCA for RNA-seq*.  
  Datasets referenced: **OSD-255**.  
  Methods observed: PCA (dimensionality reduction), clustering, EDA, XAI (SHAP where present).

- **Copy_of_explainable_ai.ipynb**  
  Headings include: *Mission of the explainable AI notebook*, *Read in data/metadata*, *Logistic regression with SHAP on RNA-seq*.  
  Datasets referenced: **OSD-255**, **OSD-583**.  
  XAI methods: `shap` (`KernelExplainer`), permutation importance; classification metrics.

- **Copy_of_bioinformatic_tools.ipynb**  
  Headings include: *Mission of the notebook*, tool overviews, PCA and XAI usage.  
  Methods observed: Dimensionality reduction (PCA), SHAP/LIME/permutation importance.

---

### Acknowledgments
Built during the **NASA TOPS-T ScienceCore** AI/ML in Space Biology training. Thanks to the open-science ecosystem (OSDR/GeneLab) enabling reproducible education & research.
