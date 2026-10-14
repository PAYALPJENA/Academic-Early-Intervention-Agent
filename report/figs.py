import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
INK = "#1f2937"; MUTED = "#6b7280"
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#9ca3af"

# Fig 1: architecture
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)
def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col+"18", ec=col, lw=1.6))
    ax.text(x+w/2, y+h-0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x+w/2, y+h-0.68, body, ha="center", va="top", fontsize=7.9, color=INK, linespacing=1.35)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
top = 2.45; h = 1.95; w = 1.95
box(0.1, top, w, h, "1. Raw Data", "500,000 students\n27 attributes\nCSV", GREY)
box(2.3, top, w, h, "2. Preprocessing", "Risk label (<40/40-69/≥70)\nDrop leakage & IDs\n+4 engineered features", BLUE)
box(4.5, top, w, h, "3. Split & Encode", "Stratified 70/15/15\n(seed 42)\nOne-Hot + Scaler\n→ 55 features,\nfitted per CV fold", BLUE)
box(6.7, top, w, h, "4. Model Training", "12 models, including\nCatBoost (selected),\nHistGradientBoosting\n5 seeds; CV where defined", ORANGE)
box(8.9, top, w, h, "5. Validate & Test", "Tune RF + thresholds\non validation; select\nTest once: argmax\nand tuned rule", ORANGE)
for x in (2.05, 4.25, 6.45, 8.65): arrow(x, top+h/2, x+0.25, top+h/2)
bot = 0.15; hb = 1.75
box(1.2, bot, 2.6, hb, "Prediction Module", "risk_prediction.py\nvalidate → engineer →\ntransform → predict_proba", GREEN)
box(4.2, bot, 2.6, hb, "Intervention Layer", "Risk-level support actions\nRule triggers (attendance,\nstress, study hours …)", GREEN)
box(7.2, bot, 2.6, hb, "Streamlit Dashboard", "23-field advisor form\nRisk card + probabilities\nHuman-in-the-loop decision", GREEN)
arrow(9.85, top, 2.5, bot+hb+0.02)
ax.text(6.6, 2.08, "CatBoost pipeline + validation thresholds; legacy HGB fallback", ha="center", va="top", fontsize=8, color=MUTED, style="italic")
arrow(3.8, bot+hb/2, 4.2, bot+hb/2); arrow(6.8, bot+hb/2, 7.2, bot+hb/2)
fig.savefig("assets/fig1_architecture.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# Fig 2: class distribution
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labs = ["High Risk\n(< 40)", "Medium Risk\n(40–69)", "Low Risk\n(≥ 70)"]; vals = [25615, 411494, 62891]; cols = [RED, ORANGE, GREEN]
b = ax.bar(labs, vals, color=cols, width=0.6)
for r, v in zip(b, vals): ax.text(r.get_x()+r.get_width()/2, v+6000, f"{v:,}\n({v/5000:.1f}%)", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_ylabel("Number of students"); ax.set_ylim(0, 480000); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))
fig.savefig("assets/fig2_class_distribution.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# Fig 3: model comparison (notebook 04, saved models on 100,000 test students)
import numpy as np
metrics = ["Accuracy", "High-Risk\nPrecision", "High-Risk\nRecall", "High-Risk\nF1", "Macro F1"]
data = {"Logistic Regression": [0.8608, 0.6901, 0.2048, 0.3158, 0.59],
        "Random Forest":       [0.8739, 0.8517, 0.2477, 0.3838, 0.62],
        "HistGradientBoosting":[0.8823, 0.7641, 0.3838, 0.5109, 0.69]}
fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(metrics)); wd = 0.26
for i, (k, v) in enumerate(data.items()):
    bb = ax.bar(x + (i-1)*wd, v, wd, label=k, color=[GREY, "#60a5fa", BLUE][i])
    for r, val in zip(bb, v): ax.text(r.get_x()+r.get_width()/2, val+0.012, f"{val:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, metrics); ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.savefig("assets/fig3_model_comparison.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# Fig 4: feature importance (Random Forest, notebook 04)
feats = [("Midterm_Mark",0.160573),("Weekly_Study_Hours",0.114396),("Previous_GPA",0.101865),("Stress_Level",0.096244),
         ("study_engagement_score*",0.086739),("Self_Efficacy_Score",0.059817),("Library_Visits_Per_Month",0.047279),
         ("Attendance_Rate",0.038871),("Motivation_Score",0.035324),("Internet_Quality",0.024684),("learning_support_score*",0.020883),("Extracurricular_Hours",0.019242)]
fig, ax = plt.subplots(figsize=(7.6, 4.2)); n=[f for f,_ in feats][::-1]; v=[s for _,s in feats][::-1]
ax.barh(n, v, color=[ORANGE if f.endswith("*") else BLUE for f in n])
for i, val in enumerate(v): ax.text(val+0.002, i, f"{val:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_xlabel("Gini importance"); ax.set_xlim(0, 0.19)
ax.text(0.185, 0.2, "* engineered feature", ha="right", fontsize=8.5, color=ORANGE)
fig.savefig("assets/fig4_feature_importance.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("done")
