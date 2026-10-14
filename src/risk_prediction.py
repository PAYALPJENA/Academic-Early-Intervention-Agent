import json
import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# =========================================================
# PATHS
# =========================================================

SELECTION_PATH = os.path.join(BASE_DIR, "results", "selection.json")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "results", "thresholds_validation.json")
OPERATING_POINTS_PATH = os.path.join(BASE_DIR, "results", "operating_points.json")
PIPELINE_MODELS_DIR = os.path.join(BASE_DIR, "models", "pipeline")
OLD_MODEL_PATH = os.path.join(BASE_DIR, "old_models", "gradient_boosting.pkl")
OLD_PREPROCESSOR_PATH = os.path.join(BASE_DIR, "old_models", "preprocessor.pkl")


# =========================================================
# LOAD THE VALIDATION-SELECTED PIPELINE (OR LEGACY FALLBACK)
# =========================================================

def _load_active_model():
    if os.path.isfile(SELECTION_PATH):
        with open(SELECTION_PATH, encoding="utf-8") as f:
            selection = json.load(f)
        selected = selection.get("tuned_threshold_rule", {})
        model_name = selected.get("selected_model")
        model_label = selected.get("label", model_name)
        if not model_name:
            raise ValueError("results/selection.json has no tuned-threshold selected model.")

        model_path = os.path.join(PIPELINE_MODELS_DIR, f"{model_name}.joblib")
        if os.path.isfile(model_path):
            with open(THRESHOLDS_PATH, encoding="utf-8") as f:
                threshold_data = json.load(f)
            try:
                seed_thresholds = threshold_data["models"][model_name]["per_seed"]["0"]
                t_high, t_low = seed_thresholds["t_high"], seed_thresholds["t_low"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"No validation-tuned seed-0 thresholds were saved for {model_name}."
                ) from exc
            model = joblib.load(model_path)
            return model, None, {
                "model_name": model_name,
                "model_label": model_label,
                "decision_rule": f"Validation-tuned thresholds (seed 0: t_high={t_high:.2f}, t_low={t_low:.2f})",
                "seed": 0,
                "path": model_path,
                "is_pipeline": True,
                "t_high": t_high,
                "t_low": t_low,
                "fallback": False,
            }

    # Keep the initial dashboard model usable if the selected pipeline artifact is absent.
    warnings.warn(
        "The selected pipeline model artifact could not be loaded; falling back to legacy HistGradientBoosting.",
        RuntimeWarning,
        stacklevel=2,
    )
    if not (os.path.isfile(OLD_MODEL_PATH) and os.path.isfile(OLD_PREPROCESSOR_PATH)):
        raise FileNotFoundError(
            "The selected pipeline model is unavailable and old_models/ does not contain the legacy fallback."
        )
    model = joblib.load(OLD_MODEL_PATH)
    prep = joblib.load(OLD_PREPROCESSOR_PATH)
    return model, prep, {
        "model_name": "legacy_histgradientboosting",
        "model_label": "Legacy HistGradientBoosting (fallback)",
        "decision_rule": "Argmax (legacy fallback)",
        "seed": None,
        "path": OLD_MODEL_PATH,
        "is_pipeline": False,
        "t_high": None,
        "t_low": None,
        "fallback": True,
    }


risk_model, legacy_preprocessor, ACTIVE_MODEL_INFO = _load_active_model()
MODEL_PATH = ACTIVE_MODEL_INFO["path"]
PREPROCESSOR_PATH = OLD_PREPROCESSOR_PATH if ACTIVE_MODEL_INFO["fallback"] else None


def get_active_model_info():
    """Return the model and decision rule currently used by the dashboard."""
    return dict(ACTIVE_MODEL_INFO)


# =========================================================
# AVAILABLE SUBJECTS
# =========================================================

AVAILABLE_SUBJECTS = [
    "Economics",
    "Architecture",
    "Data Science",
    "Law",
    "Biology",
    "Civil Engineering",
    "Business Administration",
    "Political Science",
    "Chemistry",
    "Mathematics",
    "Sociology",
    "Psychology",
    "Medicine",
    "Electrical Engineering",
    "Computer Science",
    "Physics",
    "Literature",
    "Mechanical Engineering"
]


# =========================================================
# ORIGINAL DATASET COLUMNS
# =========================================================

ORIGINAL_COLUMNS = [
    "Semester_ID",
    "Age",
    "Gender",
    "Region_Type",
    "Family_Size",
    "Major_Subject",
    "Parent_Education",
    "Family_Income_Level",
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
    "Midterm_Mark"
]


# =========================================================
# ENGINEERED FEATURES
# =========================================================

ENGINEERED_COLUMNS = [
    "previous_failure_flag",
    "study_engagement_score",
    "digital_distraction_score",
    "learning_support_score"
]


EXPECTED_COLUMNS = ORIGINAL_COLUMNS + ENGINEERED_COLUMNS


# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Shared with the training pipeline (src/pipeline/features.py)
from pipeline.features import create_engineered_features  # noqa: E402
from pipeline.models import predict_with_thresholds  # noqa: E402


# =========================================================
# PREDICT STUDENT ACADEMIC RISK
# =========================================================

def _read_operating_points():
    if not os.path.isfile(OPERATING_POINTS_PATH):
        return {}
    with open(OPERATING_POINTS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def get_sensitivity_modes():
    """Return the existing rule and validation operating points for the active model."""
    if not ACTIVE_MODEL_INFO["is_pipeline"]:
        return []
    data = _read_operating_points()
    seed = str(ACTIVE_MODEL_INFO.get("seed", 0))
    seed_points = data.get("per_seed", {}).get(seed, {})
    rule_descriptions = data.get("threshold_rules", {})
    modes = [{"key": "saved_validation", "label": "Saved validation thresholds"}]

    for rule_key, description in rule_descriptions.items():
        if rule_key not in seed_points:
            continue
        if rule_key == "f1_optimal":
            label = "Balanced (validation F1-optimal)"
        else:
            match = re.search(r"recall\s*>=\s*([0-9.]+)", description, flags=re.IGNORECASE)
            if match:
                target = float(match.group(1))
                label = f"Recall target ≥ {target:.0%}"
            else:
                label = rule_key.replace("_", " ").title()
        modes.append({"key": rule_key, "label": label})
    return modes


def get_decision_rule_info(sensitivity_mode=None):
    """Return the saved thresholds and validation operating-point metrics for a mode."""
    if not ACTIVE_MODEL_INFO["is_pipeline"]:
        return {
            "mode": "legacy_argmax",
            "label": "Legacy model argmax fallback",
            "t_high": None,
            "t_low": None,
            "validation": None,
        }
    if sensitivity_mode in (None, "saved_validation"):
        return {
            "mode": "saved_validation",
            "label": "Saved validation thresholds",
            "t_high": ACTIVE_MODEL_INFO["t_high"],
            "t_low": ACTIVE_MODEL_INFO["t_low"],
            "validation": None,
        }

    data = _read_operating_points()
    seed = str(ACTIVE_MODEL_INFO.get("seed", 0))
    try:
        point = data["per_seed"][seed][sensitivity_mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"No validation operating point {sensitivity_mode!r} was saved for seed {seed}."
        ) from exc

    labels = {mode["key"]: mode["label"] for mode in get_sensitivity_modes()}
    return {
        "mode": sensitivity_mode,
        "label": labels.get(sensitivity_mode, sensitivity_mode),
        "t_high": float(point["threshold"]),
        # Keep the saved Low-Risk fallback threshold from thresholds_validation.json.
        "t_low": ACTIVE_MODEL_INFO["t_low"],
        "validation": point.get("val", {}),
    }


def _prepare_students(students):
    if isinstance(students, pd.DataFrame):
        student_df = students.copy()
    elif isinstance(students, dict):
        student_df = pd.DataFrame([students])
    else:
        student_df = pd.DataFrame(students)

    if student_df.empty:
        raise ValueError("Provide at least one student row.")

    missing_original = [column for column in ORIGINAL_COLUMNS if column not in student_df.columns]
    if missing_original:
        raise ValueError(f"Missing required student input columns: {missing_original}")

    # Ignore unrelated CSV columns, then use the same engineered features as training.
    student_df = student_df.loc[:, ORIGINAL_COLUMNS].copy()
    student_df = create_engineered_features(student_df)
    return student_df.loc[:, EXPECTED_COLUMNS]


def predict_students_risk(students, sensitivity_mode=None):
    """Predict one or more students; probabilities are returned as fractions in [0, 1]."""
    student_df = _prepare_students(students)

    # The selected saved pipeline owns its fitted preprocessor. Only the legacy
    # fallback needs the separately saved preprocessing object.
    model_input = (
        student_df
        if ACTIVE_MODEL_INFO["is_pipeline"]
        else legacy_preprocessor.transform(student_df)
    )
    probabilities = np.asarray(risk_model.predict_proba(model_input), dtype=float)
    class_names = np.asarray(risk_model.classes_)
    if ACTIVE_MODEL_INFO["is_pipeline"]:
        rule = get_decision_rule_info(sensitivity_mode)
        predictions = predict_with_thresholds(
            probabilities,
            class_names,
            rule["t_high"],
            rule["t_low"],
        )
    else:
        predictions = class_names[np.argmax(probabilities, axis=1)]
    probability_rows = [
        {str(name): float(value) for name, value in zip(class_names, row)}
        for row in probabilities
    ]
    return [str(value) for value in predictions], probability_rows


def predict_student_risk(student_data, sensitivity_mode=None):
    predictions, probability_rows = predict_students_risk(
        [student_data], sensitivity_mode=sensitivity_mode
    )
    probabilities = {
        name: round(value * 100, 2)
        for name, value in probability_rows[0].items()
    }
    return predictions[0], probabilities


def explain_student_risk(student_data, predicted_class=None, top_n=5):
    """Explain factors that raise or lower the CatBoost High-Risk score."""
    if not ACTIVE_MODEL_INFO["is_pipeline"] or ACTIVE_MODEL_INFO["model_name"] != "catboost":
        return []

    from catboost import Pool

    student_df = _prepare_students(student_data)
    preprocessor = risk_model.named_steps["preprocessor"]
    estimator = risk_model.named_steps["model"]
    transformed = preprocessor.transform(student_df)
    shap_values = np.asarray(
        estimator.get_feature_importance(data=Pool(transformed), type="ShapValues"),
        dtype=float,
    )
    # Always explain the High-Risk class so "up" and "down" have one stable
    # meaning regardless of which class the threshold rule predicts.
    class_index = list(estimator.classes_).index("High Risk")

    if shap_values.ndim == 3:
        contributions = shap_values[0, class_index, :-1]
    elif shap_values.ndim == 2:
        contributions = shap_values[0, :-1]
    else:
        raise ValueError(f"Unexpected CatBoost SHAP shape: {shap_values.shape}")

    transformed_names = list(preprocessor.get_feature_names_out())
    if len(transformed_names) != len(contributions):
        raise ValueError("SHAP values do not match the fitted preprocessor feature names.")

    original_names = sorted(EXPECTED_COLUMNS, key=len, reverse=True)
    grouped = {}
    for transformed_name, contribution in zip(transformed_names, contributions):
        name = transformed_name.split("__", 1)[-1]
        original_name = next(
            (candidate for candidate in original_names
             if name == candidate or name.startswith(candidate + "_")),
            name,
        )
        if original_name == "Gender":
            continue
        grouped[original_name] = grouped.get(original_name, 0.0) + float(contribution)

    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    return [
        {
            "Feature": feature,
            "Contribution": value,
            "Direction": "raises High-Risk score" if value >= 0 else "lowers High-Risk score",
        }
        for feature, value in ranked
    ]


# =========================================================
# INTERVENTION RECOMMENDATIONS
# =========================================================

def generate_intervention(student_data, risk):

    # Risk-based recommendation
    if risk == "High Risk":
        recommendations = [
            "Arrange academic counselling.",
            "Provide additional subject-specific support.",
            "Create a personalized study plan.",
            "Monitor attendance and academic progress.",
            "Assign faculty mentoring or peer support.",
        ]

    elif risk == "Medium Risk":
        recommendations = [
            "Monitor academic progress regularly.",
            "Encourage consistent study habits.",
            "Set short-term academic goals.",
            "Schedule periodic faculty mentoring.",
            "Encourage the student to seek help when needed.",
        ]

    else:
        recommendations = [
            "Continue current academic practices.",
            "Maintain consistent study habits.",
            "Continue setting academic goals.",
            "Maintain healthy learning routines.",
        ]

    # Attendance
    if student_data["Attendance_Rate"] < 75:
        recommendations.append(
            "Improve class attendance and monitor attendance regularly."
        )

    # Failed courses
    if student_data["Number_of_Failed_Courses"] > 0:
        recommendations.append(
            "Provide additional support in previously failed courses."
        )

    # Study hours
    if student_data["Weekly_Study_Hours"] < 15:
        recommendations.append(
            "Increase weekly study time with a structured study plan."
        )

    # Stress
    if student_data["Stress_Level"] >= 7:
        recommendations.append(
            "Provide stress-management and student counselling support."
        )

    # Motivation
    if student_data["Motivation_Score"] < 5:
        recommendations.append(
            "Use mentoring and goal-setting activities to improve motivation."
        )

    # Social media
    if student_data["Social_Media_Usage_Hours"] > 4:
        recommendations.append(
            "Reduce excessive social-media usage during study periods."
        )

    return recommendations


# =========================================================
# SUBJECT SELECTION
# =========================================================

def select_subject():

    print("\n==============================================")
    print("        ACADEMIC EARLY-INTERVENTION AGENT")
    print("==============================================")

    print("\nAvailable Subjects:\n")

    for index, subject in enumerate(AVAILABLE_SUBJECTS, start=1):
        print(f"{index}. {subject}")

    while True:

        try:
            choice = int(
                input("\nSelect subject number: ")
            )

            if 1 <= choice <= len(AVAILABLE_SUBJECTS):
                selected_subject = AVAILABLE_SUBJECTS[choice - 1]

                print(
                    f"\nSelected Subject: {selected_subject}"
                )

                return selected_subject

            else:
                print("Invalid choice. Please select a valid number.")

        except ValueError:
            print("Please enter a number.")


# =========================================================
# GET STUDENT INPUT
# =========================================================

def get_student_input():

    subject = select_subject()

    print("\n==============================================")
    print("           ENTER STUDENT INFORMATION")
    print("==============================================")

    student = {}

    student["Semester_ID"] = int(
        input("Semester ID: ")
    )

    student["Age"] = int(
        input("Age: ")
    )

    student["Gender"] = input(
        "Gender: "
    )

    student["Region_Type"] = input(
        "Region Type: "
    )

    student["Family_Size"] = int(
        input("Family Size: ")
    )

    # Subject comes from user selection
    student["Major_Subject"] = subject

    student["Parent_Education"] = input(
        "Parent Education: "
    )

    student["Family_Income_Level"] = input(
        "Family Income Level: "
    )

    student["Internet_Quality"] = float(
        input("Internet Quality: ")
    )

    student["Study_Space_Quality"] = float(
        input("Study Space Quality: ")
    )

    student["Previous_GPA"] = float(
        input("Previous GPA: ")
    )

    student["Number_of_Failed_Courses"] = int(
        input("Number of Failed Courses: ")
    )

    student["Total_Credits_Earned"] = int(
        input("Total Credits Earned: ")
    )

    student["Weekly_Study_Hours"] = float(
        input("Weekly Study Hours: ")
    )

    student["Attendance_Rate"] = float(
        input("Attendance Rate (%): ")
    )

    student["Library_Visits_Per_Month"] = int(
        input("Library Visits Per Month: ")
    )

    student["Extracurricular_Hours"] = float(
        input("Extracurricular Hours: ")
    )

    student["Sleep_Hours"] = float(
        input("Sleep Hours: ")
    )

    student["Social_Media_Usage_Hours"] = float(
        input("Social Media Usage Hours: ")
    )

    student["Stress_Level"] = float(
        input("Stress Level: ")
    )

    student["Motivation_Score"] = float(
        input("Motivation Score: ")
    )

    student["Self_Efficacy_Score"] = float(
        input("Self Efficacy Score: ")
    )

    student["Midterm_Mark"] = float(
        input("Midterm Mark: ")
    )

    return student


# =========================================================
# DISPLAY RESULT
# =========================================================

def display_result(student, risk, probabilities):

    print("\n==============================================")
    print("              PREDICTION RESULT")
    print("==============================================")

    print(f"Subject: {student['Major_Subject']}")

    print(f"\nPredicted Academic Risk: {risk}")

    print("\nRisk Probabilities:")

    for category, probability in probabilities.items():
        print(f"{category}: {probability}%")

    print("\n==============================================")
    print("        RECOMMENDED INTERVENTIONS")
    print("==============================================")

    recommendations = generate_intervention(
        student,
        risk
    )

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):
        print(f"{number}. {recommendation}")


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("\nRisk prediction module loaded successfully!")

    print("Model:", ACTIVE_MODEL_INFO["model_label"])
    print("Decision rule:", ACTIVE_MODEL_INFO["decision_rule"])
    print("Model file:", MODEL_PATH)

    try:

        student = get_student_input()

        risk, probabilities = predict_student_risk(
            student
        )

        display_result(
            student,
            risk,
            probabilities
        )

    except Exception as error:

        print("\nError while predicting student risk:")
        print(error)
