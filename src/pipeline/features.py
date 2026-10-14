"""Feature engineering shared by training and the deployed app."""


def create_engineered_features(student_df):

    # Previous academic difficulty
    student_df["previous_failure_flag"] = (
        student_df["Number_of_Failed_Courses"] > 0
    ).astype(int)

    # Study engagement
    student_df["study_engagement_score"] = (
        student_df["Weekly_Study_Hours"]
        + student_df["Library_Visits_Per_Month"]
    )

    # Digital distraction
    student_df["digital_distraction_score"] = (
        student_df["Social_Media_Usage_Hours"]
    )

    # Learning support
    student_df["learning_support_score"] = (
        student_df["Internet_Quality"]
        + student_df["Study_Space_Quality"]
    )

    return student_df
