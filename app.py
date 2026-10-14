from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys_path = str(APP_DIR / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from risk_prediction import (  # noqa: E402
    AVAILABLE_SUBJECTS,
    ORIGINAL_COLUMNS,
    explain_student_risk,
    generate_intervention,
    get_active_model_info,
    get_decision_rule_info,
    get_sensitivity_modes,
    predict_student_risk,
    predict_students_risk,
)

st.set_page_config(
    page_title="Academic Early Intervention",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = APP_DIR / "results"
FIGURES_DIR = APP_DIR / "figures"

PAGES = [
    "Overview",
    "Student risk prediction",
    "How it works",
    "Support & motivation",
    "About the model",
    "Batch predictions",
]

FORM_DEFAULTS = {
    "Semester_ID": 2,
    "Age": 20,
    "Gender": "Female",
    "Region_Type": "Urban",
    "Family_Size": 4,
    "Major_Subject": "Economics",
    "Parent_Education": "Bachelor",
    "Family_Income_Level": "Middle",
    "Internet_Quality": 8.0,
    "Study_Space_Quality": 8.0,
    "Previous_GPA": 3.0,
    "Number_of_Failed_Courses": 0,
    "Total_Credits_Earned": 40,
    "Weekly_Study_Hours": 20.0,
    "Attendance_Rate": 85.0,
    "Library_Visits_Per_Month": 6,
    "Extracurricular_Hours": 4.0,
    "Sleep_Hours": 7.0,
    "Social_Media_Usage_Hours": 2.0,
    "Stress_Level": 4.0,
    "Motivation_Score": 8.0,
    "Self_Efficacy_Score": 8.0,
    "Midterm_Mark": 75.0,
}

EXAMPLE_PROFILES = {
    "High": {
        "Previous_GPA": 1.2,
        "Weekly_Study_Hours": 4.0,
        "Attendance_Rate": 50.0,
        "Stress_Level": 9.0,
        "Midterm_Mark": 25.0,
    },
    # The paper has High- and Low-Risk case studies. This middle profile is an
    # illustrative example using intermediate values, not a paper case study.
    "Medium": {
        "Previous_GPA": 2.5,
        "Weekly_Study_Hours": 15.0,
        "Attendance_Rate": 75.0,
        "Stress_Level": 6.0,
        "Midterm_Mark": 55.0,
    },
    "Low": {
        "Previous_GPA": 3.5,
        "Weekly_Study_Hours": 25.0,
        "Attendance_Rate": 95.0,
        "Stress_Level": 3.0,
        "Midterm_Mark": 80.0,
    },
}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metric_text(metric: dict | None) -> str:
    if not metric:
        return "—"
    return f"{metric['mean']:.3f} ± {metric.get('std', 0.0):.3f}"


def _initialize_form_state() -> None:
    for column, value in FORM_DEFAULTS.items():
        st.session_state.setdefault(f"student_{column}", value)


def _load_example(name: str) -> None:
    profile = FORM_DEFAULTS | EXAMPLE_PROFILES[name]
    for column, value in profile.items():
        st.session_state[f"student_{column}"] = value
    st.session_state["loaded_example"] = name
    st.session_state.pop("last_student", None)


def _render_sensitivity_selector() -> str | None:
    modes = get_sensitivity_modes()
    if not modes:
        model_info = get_active_model_info()
        if model_info["fallback"]:
            st.warning("The selected pipeline is unavailable; using the legacy model with argmax decisions.")
        else:
            st.warning("Saved operating points are unavailable; using the saved validation thresholds.")
        return None

    labels = {item["key"]: item["label"] for item in modes}
    mode_keys = list(labels)
    current = st.session_state.get("sensitivity_mode", mode_keys[0])
    if current not in mode_keys:
        current = mode_keys[0]
    selected = st.selectbox(
        "Decision sensitivity",
        options=mode_keys,
        index=mode_keys.index(current),
        format_func=lambda key: labels[key],
        key="sensitivity_mode",
        help="Each operating point is selected on validation data and then applied to this student's probabilities.",
    )
    return None if selected == "saved_validation" else selected


def _render_validation_operating_point(mode: str | None) -> dict:
    info = get_decision_rule_info(mode)
    validation = info.get("validation")
    if validation:
        st.caption(
            "Validation operating point: "
            f"High-Risk threshold {info['t_high']:.3f}; "
            f"precision {validation['precision']:.3f}; "
            f"recall {validation['recall']:.3f}; "
            f"flagged share {validation['flagged_share']:.1%}."
        )
    return info


def _format_model_table(rows: list[dict], selected_model: str) -> pd.DataFrame:
    metric_columns = [
        ("Test accuracy", "test_accuracy"),
        ("Macro F1", "test_macro_f1"),
        ("High-Risk precision", "test_high_risk_precision"),
        ("High-Risk recall", "test_high_risk_recall"),
        ("High-Risk F1", "test_high_risk_f1"),
        ("High-Risk PR-AUC", "test_high_risk_pr_auc"),
    ]
    records = []
    for row in rows:
        record = {
            "Selected": "✓" if row["model"] == selected_model else "",
            "Model": row["model"],
        }
        record.update({label: _metric_text(row.get(key)) for label, key in metric_columns})
        records.append(record)
    return pd.DataFrame(records)


def _show_selected_table(rows: list[dict], selected_model: str) -> None:
    table = _format_model_table(rows, selected_model)

    def highlight_selected(row):
        if row["Model"] == selected_model:
            return ["background-color: #163524; color: #e8fff0; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(table.style.apply(highlight_selected, axis=1), hide_index=True, width="stretch")


def render_about_model() -> None:
    st.title("About the model")
    st.write(
        "This page summarizes the saved experiment results. Metrics and charts are loaded from the project files; "
        "the selected validation model is highlighted."
    )

    summary_path = RESULTS_DIR / "summary_table.json"
    if not summary_path.is_file():
        st.error(f"Missing results file: {summary_path.relative_to(APP_DIR)}")
        return

    try:
        summary = _read_json(summary_path)
        table_a = summary["table_A_argmax"]
        table_b = summary["table_B_validation_tuned_thresholds"]
        selected_model = table_a["selected_on_validation"]
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read the model summary: {error}")
        return

    st.caption(f"Validation-selected model: **{selected_model}**")
    argmax_tab, tuned_tab = st.tabs(["Argmax decisions", "Validation-tuned thresholds"])
    with argmax_tab:
        st.caption("Final test metrics from results/summary_table.json, table A.")
        _show_selected_table(table_a["rows"], selected_model)
    with tuned_tab:
        st.caption("Final test metrics from results/summary_table.json, table B.")
        _show_selected_table(table_b["rows"], selected_model)

    chart_path = FIGURES_DIR / "class_distribution.png"
    st.subheader("Class distribution")
    if chart_path.is_file():
        st.image(str(chart_path), caption="Target-class distribution in the project dataset", width="stretch")
    else:
        st.warning("figures/class_distribution.png is not available.")

    st.subheader("CatBoost class-imbalance experiment")
    imbalance_path = RESULTS_DIR / "imbalance_experiment.json"
    if not imbalance_path.is_file():
        st.info("The supplementary imbalance results are not available yet.")
    else:
        try:
            experiment = _read_json(imbalance_path)
            catboost = experiment["models"]["catboost"]["settings"]
            records = []
            for setting, result in catboost.items():
                records.append(
                    {
                        "Setting": result.get("label", setting),
                        "High-Risk F1 (argmax)": _metric_text(result["test_argmax"].get("high_risk_f1")),
                        "High-Risk F1 (tuned)": _metric_text(result["test_tuned_thresholds"].get("high_risk_f1")),
                        "High-Risk PR-AUC": _metric_text(result["test_argmax"].get("high_risk_pr_auc")),
                    }
                )
            st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
            seed_count = experiment["protocol"].get("seeds", [])
            st.caption(f"Supplementary test metrics; {len(seed_count)} seeds. PR-AUC is threshold-independent.")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            st.warning(f"Could not read the imbalance summary: {error}")

    selected_row = next((row for row in table_b["rows"] if row["model"] == selected_model), None)
    if selected_row and selected_row.get("test_high_risk_recall"):
        missed_share = 1.0 - selected_row["test_high_risk_recall"]["mean"]
        missed_text = f"about {missed_share:.0%}"
    else:
        missed_text = "a material share"

    st.subheader("Limitations")
    st.markdown(
        "\n".join(
            [
                f"- At its reported validation-tuned test operating point, the selected model misses **{missed_text}** of High-Risk students.",
                "- The dataset appears likely to be synthetic or synthetic-heavy, so its patterns may not represent real students.",
                "- No fairness audit has been completed.",
                "- The model has not been validated in a real-world deployment.",
            ]
        )
    )


def render_prediction_page() -> None:
    st.title("Student risk prediction")
    st.write("Enter a student profile to see risk probabilities and possible support actions.")
    model_info = get_active_model_info()
    if model_info.get("fallback"):
        st.warning("⚠️ Legacy HistGradientBoosting fallback active. The selected CatBoost pipeline could not be loaded.")
    st.caption(f"Serving {model_info['model_label']} (seed {model_info['seed']}).")
    sensitivity_mode = _render_sensitivity_selector()

    decision_info = get_decision_rule_info(sensitivity_mode)
    _render_validation_operating_point(sensitivity_mode)

    _initialize_form_state()
    st.subheader("Load an example profile")
    button_columns = st.columns(3)
    for column, name in zip(button_columns, EXAMPLE_PROFILES):
        column.button(
            f"Load {name} risk example",
            key=f"load_example_{name.lower()}",
            use_container_width=True,
            on_click=_load_example,
            args=(name,),
        )
    if st.session_state.get("loaded_example"):
        st.caption(f"Loaded the {st.session_state['loaded_example'].lower()} example. Review the form, then analyze it.")

    with st.form("student_prediction_form"):
        st.subheader("Subject and personal information")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("Major subject", AVAILABLE_SUBJECTS, key="student_Major_Subject")
        with c2:
            st.number_input("Semester ID", min_value=1, max_value=8, key="student_Semester_ID")
        with c3:
            st.number_input("Age", min_value=17, max_value=30, key="student_Age")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("Gender", ["Female", "Male", "Non-binary"], key="student_Gender")
        with c2:
            st.selectbox("Region type", ["Urban", "Rural"], key="student_Region_Type")
        with c3:
            st.number_input("Family size", min_value=2, max_value=8, key="student_Family_Size")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox(
                "Parent education",
                ["No Education", "Primary", "Secondary", "Diploma", "Bachelor", "Postgraduate"],
                key="student_Parent_Education",
            )
        with c2:
            st.selectbox(
                "Family income level",
                ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
                key="student_Family_Income_Level",
            )
        with c3:
            st.number_input("Internet quality", min_value=1.0, max_value=10.0, step=0.5, key="student_Internet_Quality")

        st.subheader("Academic and learning information")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Study-space quality", min_value=1.0, max_value=10.0, step=0.5, key="student_Study_Space_Quality")
        with c2:
            st.number_input("Previous GPA", min_value=0.0, max_value=4.0, step=0.1, key="student_Previous_GPA")
        with c3:
            st.number_input("Failed courses", min_value=0, max_value=10, key="student_Number_of_Failed_Courses")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Total credits earned", min_value=12, max_value=144, key="student_Total_Credits_Earned")
        with c2:
            st.number_input("Weekly study hours", min_value=2.0, max_value=40.0, step=0.5, key="student_Weekly_Study_Hours")
        with c3:
            st.number_input("Attendance rate (%)", min_value=0.0, max_value=100.0, step=0.5, key="student_Attendance_Rate")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Library visits per month", min_value=0, max_value=24, key="student_Library_Visits_Per_Month")
        with c2:
            st.number_input("Extracurricular hours", min_value=0.0, max_value=30.0, step=0.5, key="student_Extracurricular_Hours")
        with c3:
            st.number_input("Sleep hours", min_value=3.0, max_value=11.0, step=0.5, key="student_Sleep_Hours")

        st.subheader("Lifestyle and well-being")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Social-media hours", min_value=0.0, max_value=12.0, step=0.5, key="student_Social_Media_Usage_Hours")
        with c2:
            st.slider("Stress level", min_value=1.0, max_value=10.0, step=0.5, key="student_Stress_Level")
        with c3:
            st.slider("Motivation score", min_value=1.0, max_value=10.0, step=0.5, key="student_Motivation_Score")

        c1, c2 = st.columns(2)
        with c1:
            st.slider("Self-efficacy score", min_value=1.0, max_value=10.0, step=0.5, key="student_Self_Efficacy_Score")
        with c2:
            st.number_input("Midterm mark", min_value=0.0, max_value=100.0, step=0.5, key="student_Midterm_Mark")

        submitted = st.form_submit_button("Analyze student risk", type="primary", use_container_width=True)

    if submitted:
        st.session_state["last_student"] = {
            column: st.session_state[f"student_{column}"] for column in ORIGINAL_COLUMNS
        }

    student = st.session_state.get("last_student")
    if not student:
        st.info("Choose an example or complete the form, then select Analyze student risk.")
        return

    try:
        with st.spinner("Scoring the student profile..."):
            prediction, probabilities = predict_student_risk(student, sensitivity_mode=sensitivity_mode)
        st.divider()
        st.subheader("Prediction result")
        st.info(f"Model: {model_info['model_label']} (seed {model_info['seed']}) · Decision mode: {decision_info['label']}.")
        if decision_info["t_high"] is None:
            st.caption("Label uses argmax: the class with the highest predicted probability.")
        else:
            st.caption(
                "Label uses validation-tuned thresholds: "
                f"High if P(High) ≥ {decision_info['t_high']:.1%}; "
                f"otherwise Low if P(Low) ≥ {decision_info['t_low']:.1%}; "
                "otherwise Medium. The operating point was chosen on validation data to favor catching High-Risk students."
            )
        st.warning("Decision support only; the advisor decides.")
        st.markdown(f"### Predicted level: **{prediction}**")

        probability_columns = st.columns(3)
        for column, risk in zip(probability_columns, ["High Risk", "Medium Risk", "Low Risk"]):
            column.metric(risk, f"{probabilities.get(risk, 0.0):.1f}%")

        ordered = sorted(probabilities.values(), reverse=True)
        if len(ordered) >= 2 and ordered[0] - ordered[1] <= 10.0:
            gap = ordered[0] - ordered[1]
            st.markdown(
                f'<span style="background:#5b3a00;color:#ffd166;padding:4px 10px;border-radius:999px;font-weight:700">Borderline</span> '
                f"Top two class probabilities are {gap:.1f} percentage points apart.",
                unsafe_allow_html=True,
            )

        probability_chart = pd.DataFrame(
            {"Risk level": list(probabilities), "Probability (%)": list(probabilities.values())}
        )
        st.bar_chart(probability_chart, x="Risk level", y="Probability (%)")

        with st.expander("Why this prediction"):
            st.caption(
                "These CatBoost SHAP values show which inputs raise or lower the model's High-Risk score. "
                "They explain the model, not the cause of a student's outcome; they are not probability points."
            )
            try:
                explanation = explain_student_risk(student, prediction, top_n=5)
                if explanation:
                    explanation_frame = pd.DataFrame(explanation)
                    explanation_frame["Feature"] = explanation_frame["Feature"].str.replace("_", " ", regex=False).str.title()
                    explanation_frame["Contribution"] = explanation_frame["Contribution"].map(lambda value: f"{value:+.4f}")
                    st.dataframe(explanation_frame, hide_index=True, width="stretch")
                else:
                    st.info("Feature explanations are unavailable for the active model.")
            except Exception as error:
                st.warning(f"Could not compute SHAP explanations: {error}")

        st.subheader("Suggested support")
        for recommendation in generate_intervention(student, prediction):
            st.markdown(f"- {recommendation}")
    except Exception as error:
        st.error("Prediction could not be completed.")
        st.exception(error)


def render_batch_page() -> None:
    st.title("Batch predictions")
    st.write("Upload a CSV containing the 23 raw input columns. Rows are ranked by predicted High-Risk probability.")
    model_info = get_active_model_info()
    if model_info.get("fallback"):
        st.warning("⚠️ Legacy HistGradientBoosting fallback active. The selected CatBoost pipeline could not be loaded.")
    sensitivity_mode = _render_sensitivity_selector()
    _render_validation_operating_point(sensitivity_mode)

    template = pd.DataFrame(columns=ORIGINAL_COLUMNS).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download 23-column CSV template",
        data=template,
        file_name="student_input_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Student CSV", type=["csv"], key="batch_student_csv")
    if uploaded is None:
        return

    try:
        students = pd.read_csv(uploaded)
        missing = [column for column in ORIGINAL_COLUMNS if column not in students.columns]
        if missing:
            st.error(f"CSV is missing required input columns: {', '.join(missing)}")
            return
        extra = [column for column in students.columns if column not in ORIGINAL_COLUMNS]
        if extra:
            st.caption(f"Ignored extra columns: {', '.join(extra)}")

        labels, probability_rows = predict_students_risk(students, sensitivity_mode=sensitivity_mode)
        output = students.loc[:, ORIGINAL_COLUMNS].copy()
        output["Predicted risk"] = labels
        for risk in ["High Risk", "Medium Risk", "Low Risk"]:
            output[f"P({risk}) (%)"] = [round(row.get(risk, 0.0) * 100, 2) for row in probability_rows]
        output = output.sort_values("P(High Risk) (%)", ascending=False).reset_index(drop=True)
        output.insert(0, "Rank", range(1, len(output) + 1))
        st.dataframe(output, hide_index=True, width="stretch")
        st.download_button(
            "Download ranked predictions",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="ranked_student_predictions.csv",
            mime="text/csv",
            key="download_batch_predictions",
        )
    except Exception as error:
        st.error(f"Batch prediction could not be completed: {error}")


def render_overview() -> None:
    st.title("Academic Early Intervention")
    st.subheader("Find students who may need support early enough to help them.")
    st.info("Every prediction is a prompt for a supportive conversation. It is not a final judgment about a student.")
    st.markdown("### How advisors use the dashboard")
    columns = st.columns(4)
    steps = [
        ("1. Enter a profile", "Use a student record or one of the example profiles."),
        ("2. Review risk", "See class probabilities, the decision rule and borderline cases."),
        ("3. Understand the score", "Review model contributions and the student's indicators."),
        ("4. Decide on support", "The advisor reviews context and chooses the next step."),
    ]
    for column, (title, body) in zip(columns, steps):
        with column:
            st.markdown(f"#### {title}")
            st.write(body)
    st.caption("The dashboard is decision support only; the advisor decides.")


def render_how_it_works() -> None:
    st.title("How it works")
    st.markdown(
        "1. The advisor enters 23 mid-semester inputs.\n"
        "2. The same four engineered indicators used by the pipeline are added.\n"
        "3. The validation-selected CatBoost pipeline estimates probabilities for High, Medium and Low Risk.\n"
        "4. A validation-selected threshold rule converts probabilities into a label.\n"
        "5. Shared intervention rules combine risk-level actions with relevant profile triggers."
    )
    st.info("SHAP values describe how this model produced a score. They do not establish what caused a student's outcome.")


def render_support() -> None:
    st.title("Support & motivation")
    st.write("A risk estimate is a reason to check in, listen, and offer support that fits the student's context.")
    st.markdown(
        "- Ask the student what support would be useful.\n"
        "- Agree on a small, specific next step.\n"
        "- Offer academic, wellbeing or mentoring resources when appropriate.\n"
        "- Revisit progress with the student rather than treating a prediction as a fixed label."
    )


with st.sidebar:
    st.markdown("## 🎓 Academic Early Intervention")
    active_model = get_active_model_info()
    st.caption(f"Selected pipeline model: {active_model['model_label']} · seed {active_model['seed']}")
    if active_model.get("fallback"):
        st.warning("⚠️ Legacy HistGradientBoosting fallback active; pipeline model unavailable.")
    page = st.radio("Navigate", PAGES, key="dashboard_page")
    st.divider()
    st.caption("Decision support only. The advisor makes the final decision.")

if page == "Overview":
    render_overview()
elif page == "Student risk prediction":
    render_prediction_page()
elif page == "How it works":
    render_how_it_works()
elif page == "Support & motivation":
    render_support()
elif page == "About the model":
    render_about_model()
elif page == "Batch predictions":
    render_batch_page()
