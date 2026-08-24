# 📱 Smartphone Addiction Prediction — Kaggle Playground Series (s6e8)

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground%20Series%20s6e8-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e8)
[![Evaluation Metric](https://img.shields.io/badge/Metric-ROC--AUC-success?style=for-the-badge)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Official Repository for the Kaggle Playground Series Season 6 Episode 8 Challenge.**
> This repository is dedicated to exploring, modeling, and predicting smartphone addiction probability (`addicted_label`) using tabular demographic, usage, and lifestyle behavioral data.

---

## 📌 About The Project

Welcome to the **2026 Kaggle Playground Series (s6e8)** challenge! The goal of this project is to build machine learning models to predict smartphone addiction based on synthetic datasets derived from real-world smartphone behavioral research.

This repository serves as a centralized hub for:
* **Exploratory Data Analysis (EDA)** to understand digital behavior patterns.
* **Feature Engineering** experiments on screen time, notification frequency, and psychological impact indicators.
* **Machine Learning Pipelines** utilizing tabular modeling techniques.
* **Submission Management** for predicting target probabilities evaluated via ROC-AUC.

---

## 🎯 Competition Details

### 🏆 Goal
Predict the probability of the target variable **`addicted_label`** for each unique sample `id` in the test set.

### 📐 Evaluation Metric
Submissions are evaluated on **Area Under the ROC Curve (ROC-AUC)** between predicted probabilities and the observed target label.

### 📅 Timeline
* **Start Date:** August 1, 2026
* **Final Submission Deadline:** August 31, 2026 (11:59 PM UTC)

---

## 📊 Dataset Description

The dataset (both `train.csv` and `test.csv`) was synthetically generated from the *Smartphone Addiction Prediction Dataset*, capturing key user habits, demographics, and self-reported impact metrics.

### Files
* `train.csv`: The training set containing feature columns and the target variable (`addicted_label`).
* `test.csv`: The test set containing feature columns without target labels.
* `sample_submission.csv`: A sample submission file formatted with `id` and `addicted_label`.

### Features Breakdown

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `Integer` | Unique identifier for each individual record. |
| `age` | `Numerical` | Age of the individual in years. |
| `daily_screen_time_hours` | `Numerical` | Total daily phone usage duration in hours. |
| `social_media_hours` | `Numerical` | Daily hours spent on social media platforms. |
| `gaming_hours` | `Numerical` | Daily hours spent on mobile gaming. |
| `work_study_hours` | `Numerical` | Daily hours spent using smartphone for work or academic study. |
| `sleep_hours` | `Numerical` | Average daily sleep duration in hours. |
| `notifications_per_day` | `Numerical` | Average number of push notifications received per day. |
| `app_opens_per_day` | `Numerical` | Average number of times applications are opened per day. |
| `weekend_screen_time` | `Numerical` | Daily screen time recorded during weekends (in hours). |
| `gender` | `Categorical` | Gender of the individual. |
| `stress_level` | `Numerical / Ordinal` | Self-reported stress level score. |
| `academic_work_impact` | `Numerical / Ordinal` | Perceived impact of smartphone usage on work/academics. |
| **`addicted_label`** | **`Binary Target`** | **Target variable: Probability score indicating smartphone addiction.** |

---

## 📂 Repository Structure

```text
smartphone_addicting/
├── data/
│   ├── raw/                   # Original dataset files (train.csv, test.csv, sample_submission.csv)
│   └── processed/             # Preprocessed feature sets
├── notebooks/                 # Jupyter notebooks for EDA and model experimentation
├── src/                       # Source code for data processing and model pipelines
├── submissions/               # Generated submission files
├── requirements.txt           # Python dependencies
├── LICENSE                    # Project license
└── README.md                  # Project overview and documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/smartphone_addicting.git
cd smartphone_addicting
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Competition Data

Place the dataset files in `data/raw/` using the Kaggle API:

```bash
kaggle competitions download -c playground-series-s6e8 -p data/raw/
unzip data/raw/playground-series-s6e8.zip -d data/raw/
```

### 4. Training Models & Saving Checkpoints

Running `train.py` automatically trains the 10-fold cross-validation ensemble (LightGBM, XGBoost, CatBoost, TabM / Neural Net, and Ridge Meta-Learner), saves all best model artifacts to `models/`, and generates `submissions/submission.csv`:

```bash
# Train models & save best checkpoints to models/
python src/models/train.py --folds 10

# Train in Kaggle Notebook specifying custom input/output directories
python src/models/train.py --data-dir /kaggle/input/playground-series-s6e8 --model-dir /kaggle/working/models --output-dir /kaggle/working/submissions
```

### 5. Standalone Inference on `test.csv`

To run inference on any `test.csv` file using previously saved model checkpoints:

```bash
# Run standalone inference using saved checkpoints in models/
python src/models/predict.py --test-path /path/to/test.csv --output-dir submissions/
```

### 6. Submission Format Requirements

Submissions are generated in the exact format required by Kaggle (`id,addicted_label`):

```csv
id,addicted_label
691369,0.269137
691370,0.369137
691371,0.100000
```

---

## 📚 Citation

If you use this dataset or reference this competition, please use the following citation:

```bibtex
@misc{playground-series-s6e8,
    author = {Yao Yan, Walter Reade, Elizabeth Park},
    title = {Predicting Smartphone Addiction},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s6e8}},
    note = {Kaggle}
}
```

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
