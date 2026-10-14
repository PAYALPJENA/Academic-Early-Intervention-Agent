"""Single source of truth for paths, labels, columns, split and seeds."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DATA_PATH = os.path.join(ROOT, "data", "raw", "Student Performance and Behaviour.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
MODELS_DIR = os.path.join(ROOT, "models", "pipeline")


# =========================================================
# TARGET (unchanged from notebooks/02_preprocessing.ipynb)
# =========================================================

TARGET_SOURCE = "Final_Exam_Score"
TARGET = "Academic_Risk"

# score < 40 -> High Risk, 40 <= score < 70 -> Medium Risk, score >= 70 -> Low Risk
HIGH_RISK_BELOW = 40
LOW_RISK_FROM = 70

CLASS_ORDER = ["High Risk", "Medium Risk", "Low Risk"]

# Class counts the labelling must reproduce on the full raw dataset.
# The pipeline stops if they ever differ, so labels cannot drift silently.
EXPECTED_CLASS_COUNTS = {"High Risk": 25615, "Medium Risk": 411494, "Low Risk": 62891}


# =========================================================
# COLUMNS (unchanged from notebooks/02_preprocessing.ipynb)
# =========================================================

DROP_COLUMNS = ["Student_ID", "University_Name", "Home_City"]

CATEGORICAL_COLUMNS = [
    "Major_Subject",
    "Gender",
    "Region_Type",
    "Parent_Education",
    "Family_Income_Level",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Family_Size",
    "Internet_Quality",
    "Study_Space_Quality",
    "Previous_GPA",
    "Number_of_Failed_Courses",
    "Total_Credits_Earned",
    "Weekly_Study_Hours",
    "Attendance_Rate",
    "Library_Visits_Per_Month",
    "Extracurricular_Hours",
    "Sleep_Hours",
    "Social_Media_Usage_Hours",
    "Stress_Level",
    "Motivation_Score",
    "Self_Efficacy_Score",
    "Midterm_Mark",
    "previous_failure_flag",
    "study_engagement_score",
    "digital_distraction_score",
    "learning_support_score",
]


# =========================================================
# SPLIT, CV AND SEEDS
# =========================================================

SPLIT_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

CV_FOLDS = 5
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]

# Random Forest tuning grid, scored on the validation set only
RF_TUNING_SEED = 42
RF_TUNING_GRID = {
    "max_depth": [12, 16, 20],
    "min_samples_leaf": [5, 10, 20],
}

# Midterm-only Logistic Regression baseline, tuned on the validation set only
MIDTERM_LR_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}
# Decision thresholds on P(High Risk) and P(Low Risk), searched on validation probabilities
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]

# Added models: light grids, each scored on the validation set only (fit seed = RF_TUNING_SEED)
EXTRA_TUNING_GRIDS = {
    "xgb": {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "n_estimators": [300]},
    "lgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1], "n_estimators": [300]},
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "iterations": [500]},
    "mlp": {"hidden": [128, 256], "lr": [1e-3, 3e-3], "n_layers": [2]},
    "ft_transformer": {"lr": [3e-4, 1e-3], "d_token": [64], "n_blocks": [2]},
}
# Deep models: the number of epochs is also chosen on validation (best epoch up to this maximum)
DEEP_MAX_EPOCHS = {"mlp": 30, "ft_transformer": 10}
# Deep models are trained on a stratified subsample of the training set (CPU time; see results/config.json)
DEEP_TRAIN_SUBSAMPLE = 100_000

# Model selection uses the validation set only: High-Risk F1, ties broken by macro F1
SELECTION_METRIC = "high_risk_f1"
SELECTION_TIEBREAK = "macro_f1"
