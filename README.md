# NASA TOPS-T ScienceCore: AI/ML in Space Biology — Notebook Suite

## Overview
This notebook suite was completed as part of the **NASA Transform to Open Science (TOPS-T) – ScienceCore: AI/ML in Space Biology Training**.  
It walks from data fundamentals → visualization/EDA → tabular ML (classification & regression) → image data → clustering/unsupervised learning → explainable AI (XAI), with emphasis on open-science practices and NASA space-biology datasets (OSDR / GeneLab).

---

## Notebook Index (what each file covers)

- **Introduction to Data**  
  *Intro to data types & metadata; context from space-biology use cases (e.g., SANS).*

- **Data Visualization**  
  *Exploratory plots, distributions, pairplots, correlation heatmaps; communicating results.*

- **Tabular Data**  
  *Loading/cleaning tabular data; pandas DataFrames; basic feature engineering.*

- **Classification**  
  *Supervised learning for discrete labels (train/test split, metrics, model comparison).*

- **Regression**  
  *Supervised learning for continuous targets (error metrics, baselines, regularization).*

- **Image Data**  
  *Bioimaging pipelines: reading, preprocessing, and basic deep-learning scaffolding.*

- **Clustering**  
  *Unsupervised learning; dimensionality reduction (e.g., PCA), clustering & evaluation.*

- **Explainable AI**  
  *XAI for space-bio signals: SHAP/LIME, permutation importance, partial dependence.*

- **Bioinformatic Tools**  
  *Space-bio tooling overview; dimensionality reduction; ties to downstream ML/XAI.*

---

## Space Biology Topics Covered

- **Spaceflight vs. Ground Control comparisons (rodent models)**  
  Using OSDR/GeneLab datasets to contrast flight and ground samples, especially in the eye/retina context (GLDS/OSD-255 RNA-seq; metadata labels “Space Flight” vs “Ground Control”).

- **Spaceflight-Associated Neuro-ocular Syndrome (SANS)**  
  Intro and background on SANS, its relevance to astronaut vision, and why ocular/retinal endpoints matter in spaceflight studies.

- **Rodent Research-9 (RR9) mission focus**  
  RR9 mission context and data usage; analyses built around RR9 ocular outcomes and retinal tissue.

- **Retina/ocular phenotyping**
  - **Tonometry / Intraocular Pressure (IOP)** measures (OSD-583).
  - **Immunostaining microscopy (PECAM/CD31)** as an endothelial/vascular marker (OSD-568).
  - **Tomography/ophthalmic imaging** references (OSD-557).  
    These phenotypes are linked to flight status and used as targets/labels in ML.

- **Transcriptomics under spaceflight (RNA-seq)**
  - Normalized gene-expression matrices from **GLDS-255 / OSD-255** (mouse retina).
  - Differential expression filtering (DE/“DGEA”) to enrich flight vs. ground signals.
  - Integrating RNA-seq with phenotypes (e.g., predicting PECAM or thresholds from transcriptomic features).

- **Biological pathways & signatures**  
  Pathway-level interpretation (e.g., MSigDB inputs) suggesting ties to **vascular/angiogenesis** and **muscle/structure** phenotypes relevant to microgravity adaptation.

- **Unsupervised signatures of spaceflight**
  - **PCA** visualizations separating flight/ground samples.
  - **k-means clustering** on RNA-seq (with outlier checks), highlighting whether flight status forms distinct molecular clusters.

- **Explainable models connecting genes → ocular phenotypes**
  - **Linear/Ridge regression** and **Random Forest** models that map RNA-seq to **PECAM** microscopy readouts.
  - **SHAP** and **permutation importance** to identify **gene drivers** most associated with ocular/vascular endpoints under spaceflight.

- **Bioimaging data handling**  
  Basic pipelines for microscopy/image data (loading, preprocessing), with ties to molecular data where possible.

---

## Datasets Used

NASA Open Science Data Repository (OSDR) / GeneLab references:
- **OSD-255** (NASA OSDR)
- **OSD-557** / **OSDR-557** (NASA OSDR)
- **OSD-568** (NASA OSDR)
- **OSD-583** (NASA OSDR)

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
- Linear models like **Perceptron** / **SGDClassifier**
- Model selection: **GridSearchCV** / **RandomizedSearchCV**

### Supervised learning — Regression
- **LinearRegression**, **Ridge**, **Lasso** (regularization)
- Tree-based/ensemble regressors when included
- **SVR** / **SGDRegressor**

### Unsupervised learning
- **KMeans**
- **silhouette_score** (cluster quality)
- PCA projections for visualization of clusters

### Deep learning
- Framework scaffolding with **PyTorch** and **Keras/TensorFlow**
- Basic image preprocessing (PIL / scikit-image)

### Explainable AI (XAI)
- **SHAP**
- **LIME**
- **Permutation importance**
- **Partial dependence**

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

### Acknowledgments
Built during the **NASA TOPS-T ScienceCore** AI/ML in Space Biology training. Thanks to the open-science ecosystem (OSDR/GeneLab) enabling reproducible education & research.
