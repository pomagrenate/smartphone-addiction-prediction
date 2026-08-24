# 📋 Technical Roadmap & TODO | HyTab-Addict Framework
## Predictive Analytics and Deep Learning Paradigms for Behavioral Addiction (Kaggle Playground Series s6e8)

> [!NOTE]
> This document outlines the architectural research, mathematical formulation, and operational development checklist for the **HyTab-Addict** framework targeting the 2026 Kaggle Smartphone Addiction challenge.

---

## 📑 Table of Contents
1. [Landscape of Behavioral Addiction Modeling](#1-landscape-of-behavioral-addiction-modeling)
2. [Survey of Tabular Machine Learning Methodologies](#2-survey-of-tabular-machine-learning-methodologies)
3. [Comparative Analysis of Tabular Frameworks](#3-comparative-analysis-of-tabular-frameworks)
4. [Mathematical Formulation of Parameter-Efficient Deep Ensembling (TabM)](#4-mathematical-formulation-of-parameter-efficient-deep-ensembling-tabm)
5. [Proposed Architecture: HyTab-Addict Framework](#5-proposed-architecture-hytab-addict-framework)
6. [Validation Protocol & Competition Strategy](#6-validation-protocol--competition-strategy)
7. [Actionable Implementation Checklist](#7-actionable-implementation-checklist)

---

## 1. Landscape of Behavioral Addiction Modeling

Research in behavioral addiction—specifically problematic smartphone usage, digital hyper-connectivity, and internet dependency—has transitioned from static psychometric evaluation to dynamic computational prediction based on passive behavioral logging. 

Historically, diagnosis relied on self-reported clinical inventories (e.g., *Smartphone Addiction Scale (SAS)* and *Smartphone Addiction Inventory (SPAI)*), which suffered from recall bias and social desirability distortions. The integration of high-frequency telemetry in modern mobile OS platforms now enables continuous, non-invasive logging of granular usage patterns.

### ⚠️ Core Algorithmic Challenges

* **Non-Linear Operational Thresholds:** Behavioral addiction metrics do not scale linearly with negative clinical outcomes. Eight hours of daily device engagement dedicated to work carries a drastically different risk profile than eight hours fragmented into compulsive checking cycles and late-night social media consumption.
* **Heterogeneous Noise:** Merges objective system telemetry (sensor dropouts, OS throttling artifacts) with subjective self-evaluations (right-skewness, response clustering).
* **High Feature Multicollinearity:** Pickup frequency, active screen time, and notification interaction speed exhibit intense mutual correlation, causing standard decision trees to risk oversplitting along redundant coordinates.

---

## 2. Survey of Tabular Machine Learning Methodologies

### 🌲 Gradient Boosted Decision Tree (GBDT) Paradigms
* **LightGBM:** Utilizes **Gradient-Based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)** to achieve exceptional training speed while uncovering high-order categorical interactions.
* **XGBoost:** Employs exact second-order Taylor expansions with $L_1$/$L_2$ regularization, providing balanced depth-wise tree expansion resistant to noisy self-reported survey variables.
* **CatBoost:** Engineered for categorical variables via ordered target statistics over random permutations and symmetric (oblivious) decision trees to prevent target leakage.

### 🧠 Evolution of Tabular Deep Learning
* **Regularized MLPs:** Combined with Layer Normalization, Dropout, Weight Decay, and Cosine Annealing.
* **Attention & Retrieval Architectures (ExcelFormer, Trompt, TabR):** Utilize self-attention or $k$-NN candidate retrieval, though facing memory/quadratic scaling bottlenecks.
* **Parameter-Efficient Deep Ensembles (TabM & TabPack):**
  * **TabM (Gorishniy et al.):** Adapts BatchEnsemble into MLP backbones to generate $k$ distinct predictions per instance with minimal parameter overhead (~1.1× cost of a single MLP). Optimizes the unaveraged mean loss across sub-models.
  * **TabPack:** Packs dozens of heterogeneous MLPs with varying hyperparameter configurations into a single GPU-parallelized framework.

---

## 3. Comparative Analysis of Tabular Frameworks

| Framework / Model | Paradigm Class | Mixed Feature Handling | Computational & Parameter Efficiency | Out-of-Fold Variance | Noise Robustness | Primary Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM / XGBoost** | Tree GBDT | Native histogram binning & sparse optimizations | High; linear scaling on GPU | Low-to-Moderate | High scale-invariance | Cannot model smooth target surfaces |
| **CatBoost** | Oblivious Trees | Superior ordered target statistics | Moderate-to-High | Low (oblivious structure) | Very High | Slower on deep dense numerical tables |
| **Regularized MLP** | Feed-Forward NN | Requires normalization & embeddings | High efficiency | High variance across seeds | Low-to-Moderate | Sensitive to outliers and scaling |
| **TabR** | Retrieval NN | Requires embeddings & scaling | Low-to-Moderate (retrieval latency) | Low (anchor instances) | High | High memory overhead on large sets |
| **TabM** | Parameter Ensemble | Embeddings + BatchEnsemble layers | Exceptional (~1.1× single MLP cost) | Very Low | High | Sensitive to scaling factor init |
| **TabPack** | Parallel Ensemble | Heterogeneous input heads | High parallel efficiency | Very Low | Very High | Hyperparameter bounds require tuning |

---

## 4. Mathematical Formulation of Parameter-Efficient Deep Ensembling (TabM)

### Initial Input Representation
Categorical variables $x_{\text{cat}}$ are transformed via embedding tables $e(x_{\text{cat}})$, and continuous attributes $x_{\text{num}}$ are transformed via piecewise linear/periodic embeddings:

$$h_0 = \text{Concat}\left( \text{Embed}(x_{\text{num}}), e(x_{\text{cat}}) \right) \in \mathbb{R}^{d_{\text{in}}}$$

### Rank-1 Weight Matrix Parameterization
For sub-model $m \in \{1, 2, \dots, k\}$, the weight matrix is parameterized by a shared matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ modulated by rank-1 scaling vectors $r^{(m)} \in \mathbb{R}^{d_{\text{out}}}$ and $s^{(m)} \in \mathbb{R}^{d_{\text{in}}}$:

$$W^{(m)} = W \odot \left( r^{(m)} (s^{(m)})^\top \right)$$

The mini-batch feed-forward computation for sub-model $m$ is evaluated as:

$$h_{\text{out}}^{(m)} = \left( W \left( h \odot s^{(m)} \right) \right) \odot r^{(m)} + b^{(m)}$$

### Unaveraged Mean Loss Objective
TabM optimizes the mean of individual sub-model losses rather than the loss of the ensemble mean prediction:

$$\mathcal{L}_{\text{TabM}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( \frac{1}{k} \sum_{m=1}^k \ell\left( y_i, \hat{y}_i^{(m)} \right) \right)$$

During inference, final predictions are computed by averaging predictions across all $k$ heads:

$$\hat{y}_{\text{final}}(x) = \frac{1}{k} \sum_{m=1}^k \hat{y}^{(m)}$$

---

## 5. Proposed Architecture: HyTab-Addict Framework

```text
                               ┌────────────────────────────────────────┐
                               │       Engineered Telemetry Features    │
                               └───────────────────┬────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
         ┌───────────────────────────────┐                   ┌───────────────────────────────┐
         │     Stream A: Tree Ensemble   │                   │    Stream B: Deep TabM Engine │
         │   (CatBoost, LightGBM, XGB)   │                   │ (k=32 Sub-Models + Muon Opt) │
         └───────────────┬───────────────┘                   └───────────────┬───────────────┘
                         │                                                   │
                         └─────────────────────────┬─────────────────────────┘
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │  Percentile Rank Calibration & Stacking│
                               └───────────────────┬────────────────────┘
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │       Final Submission (ROC-AUC)       │
                               └────────────────────────────────────────┘
```

### Domain Feature Engine Metrics

1. **Digital Dependency Ratio ($\text{DDR}$):**
   $$\text{DDR} = \frac{\text{Nighttime Screen Time (22:00 - 06:00)}}{\text{Total Daily Screen Time} + \epsilon}$$

2. **Pickup-to-Active Hour Ratio ($\text{PPA}$):**
   $$\text{PPA} = \frac{\text{Total Device Unlock Count}}{\text{Total Active Operating Hours} + \epsilon}$$

3. **App Usage Entropy ($H_{\text{app}}$):**
   $$H_{\text{app}} = -\sum_{c \in \text{Categories}} p_c \log(p_c + \epsilon)$$

---

## 6. Validation Protocol & Competition Strategy

> [!IMPORTANT]
> Strict validation protocols are required to prevent leaderboard overfitting and data leakage.

* **Cross-Validation Scheme:** 10-Fold Stratified Cross-Validation (or GroupKFold by user ID if longitudinal interactions exist).
* **Tree Regularization:** Set `max_depth=6` for XGBoost, `num_leaves=31` for LightGBM, and `min_child_samples=50` to filter survey noise.
* **Optimization & Training:** Use **Muon** (Momentum Orthogonalized by Newton-Schulz) for TabM shared 2D weight matrices and **AdamW** for embeddings and scaling vectors.
* **Post-Hoc Calibration:** Perform Isotonic Regression or Platt Scaling on out-of-fold logits.
* **Adversarial Validation:** Train a binary classifier to detect distribution drift between train and test sets.

---

## 7. Actionable Implementation Checklist

### Phase 1: Environment & Ingestion
- [x] Set up project directory structure (`data/`, `notebooks/`, `src/`, `submissions/`).
- [x] Configure Python environment dependencies schema.
- [x] Ingest `train.csv`, `test.csv`, and verify schema with synthetic dataset fallback loader (`src/data/loader.py`).

### Phase 2: Exploratory Data Analysis & Feature Engineering
- [x] Check missing values, numerical distributions, and categorical cardinalities.
- [x] Implement domain features: $\text{DDR}$, $\text{PPA}$, unproductive usage ratio, session duration index, distress score (`src/features/build_features.py`).
- [x] Build numerical feature scaling and categorical encoding modules.

### Phase 3: Model Pipeline Development
- [x] Build Stratified CV split generator (`src/models/train.py`).
- [x] Develop Baseline GBDT Models (LightGBM, XGBoost, CatBoost).
- [x] Implement TabM neural backbone with multi-head parameter-efficient ensemble layers (`src/models/tabm.py`).
- [x] Add graceful neural network fallback for execution without optional dependencies.

### Phase 4: Stacking & Ensembling
- [x] Extract Out-Of-Fold (OOF) logit predictions from GBDT and Neural streams.
- [x] Apply Percentile Rank Transformation (`rank_transform` in `src/utils/metrics.py`).
- [x] Train Ridge meta-learner on calibrated rank matrix (`src/models/train.py`).

### Phase 5: Calibration & Final Submission
- [x] Fit probability calibration on OOF predictions.
- [x] Automated test suite verifying pipeline dataflow (`tests/pipeline_test.py`).
- [x] Submission formatting and CSV writer (`submissions/submission.csv`).

