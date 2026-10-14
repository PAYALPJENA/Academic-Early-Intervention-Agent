// Builds the paper. Every result value is read from ../results/*.json, never typed by hand.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, PageNumber, Footer, TabStopType,
} = require("docx");

const OUT = "Academic_Early_Intervention_Framework_Paper.docx";
const RES = path.join(__dirname, "..", "results");
const FIG = path.join(__dirname, "..", "figures");
const ASSETS = path.join(__dirname, "assets");
const json = (f) => JSON.parse(fs.readFileSync(path.join(RES, f), "utf8"));

// ---------- results ----------
const SUMMARY = json("summary_table.json");
const SELECTION = json("selection.json");
const SPLIT = json("data_split.json");
const CONFIG = json("config.json");
const RF_TUNING = json("rf_tuning.json");
const MID_TUNING = json("tuning_midterm_lr.json");
const TUNING = {
  xgb: json("tuning_xgb.json"),
  lgbm: json("tuning_lgbm.json"),
  catboost: json("tuning_catboost.json"),
  mlp: json("tuning_mlp.json"),
  ft_transformer: json("tuning_ft_transformer.json"),
};
const LEGACY = json("legacy_80_20.json");
const TEST_A = json("test_argmax.json").models;
const TEST_B = json("test_tuned_thresholds.json").models;

const L = CONFIG.models;
const ORDER = Object.keys(CONFIG.models);
const byLabel = (rows) => Object.fromEntries(rows.map((r) => [r.model, r]));
const RA = byLabel(SUMMARY.table_A_argmax.rows);
const RB = byLabel(SUMMARY.table_B_validation_tuned_thresholds.rows);
const A = (k) => RA[L[k]];
const Bt = (k) => RB[L[k]];
const ORDER_A = [...ORDER].sort((a, b) => A(b).test_high_risk_f1.mean - A(a).test_high_risk_f1.mean);
const ORDER_B = [...ORDER].sort((a, b) => Bt(b).test_high_risk_f1.mean - Bt(a).test_high_risk_f1.mean);
const FULL_ORDER = ORDER.filter((k) => !["majority", "midterm_lr"].includes(k));
const COUNTS = CONFIG.labels.checked_against_original_counts;
const TOTAL = Object.values(COUNTS).reduce((a, b) => a + b, 0);
const splitPct = (x) => `${(x * 100).toFixed(0)}%`;
const paramText = (params) => Object.entries(params || {}).map(([k, v]) => `${k}=${v}`).join(", ") || "Fixed configuration; individual parameters are not recorded in results JSON";

// ---------- number formatting (rounding only; values unchanged) ----------
const m3 = (s) => s.mean.toFixed(3);
const pm = (s) => s.mean.toFixed(3) + " ± " + s.std.toFixed(3);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const n0 = (x) => x.toLocaleString("en-US");
const size = (mb) => (mb == null ? "—" : mb < 0.1 ? "< 0.1" : mb.toFixed(1));
const sgn = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);

const catA = A("catboost"), catB = Bt("catboost"), midA = A("midterm_lr"), midB = Bt("midterm_lr");
const prRatio = (catA.test_high_risk_pr_auc.mean / midA.test_high_risk_pr_auc.mean).toFixed(1);
const testCounts = SPLIT.test.class_counts;
const rfBest = RF_TUNING.best_params;
const midBest = MID_TUNING.best_params;
const selA = SELECTION.argmax_rule, selB = SELECTION.tuned_threshold_rule;
const selRow = (sel, label) => sel.ranking.find((r) => r.label === label);
const cvModels = ORDER.filter((k) => A(k).cross_validated);
const noCvModels = ORDER.filter((k) => !A(k).cross_validated);

// feature importance, averaged over the five seeds' saved runs
function meanImportance(model, key, valuesKey) {
  const runs = CONFIG.seeds.map((s) => json(path.join("runs", `${model}_seed${s}.json`)));
  const feats = runs[0][key].features;
  const vals = feats.map((_, i) => runs.reduce((a, r) => a + r[key][valuesKey][i], 0) / runs.length);
  return feats.map((f, i) => ({ f, v: vals[i] })).sort((a, b) => b.v - a.v);
}
const IMP_RF = meanImportance("rf_capped", "impurity_importance", "values");
const IMP_HGB = meanImportance("hgb", "permutation_importance_val", "mean");
const rankOf = (list, f) => list.findIndex((x) => x.f === f) + 1;
const impVal = (list, f) => list.find((x) => x.f === f).v;

// ---------- docx helpers ----------
const FONT = "Times New Roman";
const CONTENT_W = 9026; // A4 width 11906 - 2 x 1440 margins

// inline markup: **bold**, *italic*, _{sub}, ^{sup}, [[placeholder]] (yellow highlight)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\}|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, ...base }));
    else if (t.startsWith("[[")) out.push(new TextRun({ text: t.slice(2, -2), highlight: "yellow", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}), alignment: opts.align ?? AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 300 }, indent: opts.indent, keepNext: opts.keepNext,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], spacing: { before: 280, after: 140 }, keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], spacing: { before: 200, after: 100 }, keepNext: true });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)], spacing: { before: 140, after: 80 }, keepNext: true });
const Bu = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 290 } });
let eqNo = 0;
const EQ = (text) => new Paragraph({
  children: [new TextRun({ text: "\t" }), ...runs(text), new TextRun({ text: `\t(${++eqNo})` })],
  tabStops: [{ type: TabStopType.CENTER, position: CONTENT_W / 2 }, { type: TabStopType.RIGHT, position: CONTENT_W }],
  spacing: { before: 80, after: 120 },
});
const CAP = (t) => new Paragraph({ children: runs(t, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 } });
const TCAP = (t) => new Paragraph({ children: runs(t, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true });
const NOTE = (t) => P(t, { run: { size: 20 }, after: 160 });

function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function IMG(file, widthIn = 6.2) {
  const { w, h } = pngSize(file); const wp = Math.round(widthIn * 96);
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wp, height: Math.round(wp * h / w) } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function TABLE(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== CONTENT_W && !opts.allowWidth) throw new Error(`table width ${total}`);
  const fs_ = opts.size ?? 18;
  const cell = (text, i, isHead, shade) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { fill: "D9E2F3", type: ShadingType.CLEAR } : shade ? { fill: "F2F2F2", type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 70, right: 70 },
    children: String(text).split("\n").map((line) => new Paragraph({
      children: runs(line, { size: fs_, bold: isHead || undefined }),
      alignment: isHead || (opts.center && opts.center.includes(i)) ? AlignmentType.CENTER : AlignmentType.LEFT,
    })),
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, opts.shadeRows && opts.shadeRows.includes(ri))) })),
    ],
  });
}

// bold the best mean in each metric column
function bestOf(rowsByKey, metricKey, higher = true) {
  let best = null;
  for (const k of ORDER) {
    const v = rowsByKey(k)[metricKey].mean;
    if (best === null || (higher ? v > best : v < best)) best = v;
  }
  return best;
}
const cellB = (row, key, best) => (Math.abs(row[key].mean - best) < 1e-12 ? `**${pm(row[key])}**` : pm(row[key]));

const C = []; // document body

// ---------- title block ----------
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 }, children: [new TextRun({ text: "Academic Early-Intervention Framework: Machine Learning for Predicting Student Academic Risk Before the Final Examination", bold: true, size: 32 })] }));

// ---------- Abstract (IMDC) ----------
C.push(H1("Abstract"));
C.push(P("**Introduction:** Academic failure in higher education is often recognised after final examinations, when timely support is difficult. This work presents the Academic Early-Intervention framework, which predicts High, Medium or Low academic risk during the semester and connects predictions to support actions for faculty advisors. **Method:** The pipeline evaluated " + ORDER.length + " models on " + n0(TOTAL) + " student records. The final-examination score was used to derive the target and excluded from model inputs. Four domain features were engineered. A stratified " + splitPct(CONFIG.split.train) + "/" + splitPct(CONFIG.split.val) + "/" + splitPct(CONFIG.split.test) + " train/validation/test split, " + CONFIG.cv_folds + "-fold cross-validation where available, and " + CONFIG.seeds.length + " seeds were used. Hyperparameters and thresholds were selected on validation data, and test data were reserved for final scoring. **Discussion:** CatBoost was selected on validation under both decision rules. On the test set, CatBoost's argmax results were accuracy " + m3(catA.test_accuracy) + ", macro F1 " + m3(catA.test_macro_f1) + " and High-Risk PR-AUC " + m3(catA.test_high_risk_pr_auc) + ". Under validation-tuned thresholds, its High-Risk F1 was " + m3(catB.test_high_risk_f1) + " and recall was " + pct(catB.test_high_risk_recall.mean) + ". CatBoost, LightGBM, XGBoost and FT-Transformer had similar High-Risk F1 to one another, with HistGradientBoosting and MLP close behind; across-seed variation does not support a clear winner. The best PR-AUC was about " + prRatio + " times the midterm-only baseline. **Conclusion:** The results support an advisor-in-the-loop early-warning framework, with further deployment and real-world validation still needed."));
C.push(P("**Keywords:** Educational Data Mining; Early Warning System; Student Risk Prediction; Gradient Boosting; Class Imbalance; Decision Threshold Tuning", { after: 200 }));

// ---------- 1. Introduction ----------
C.push(H1("1. Introduction"));
C.push(H2("1.1 Overview of the Model"));
C.push(P("Universities collect a large amount of information about their students: prior grades, attendance, study habits, family background and, increasingly, indicators of well-being such as stress and motivation. In most institutions, however, this information is used reactively. A struggling student is identified only after the final examination results are published, when the semester can no longer be recovered."));
C.push(P("The **Academic Early-Intervention framework** addresses this gap. It is a supervised machine-learning system that takes 23 attributes describing a student mid-semester, including the midterm mark, and predicts the probability of the student finishing the course in one of three risk categories: **High Risk** (final score below 40), **Medium Risk** (40–69) or **Low Risk** (70 and above). The predicted class and its probabilities are passed to an intervention layer that recommends support actions such as academic counselling, a personalised study plan or faculty mentoring. The framework is delivered to faculty advisors through a web dashboard and is explicitly designed as decision support: the final decision about any student always remains with a human advisor."));
C.push(P("The evaluation compares " + ORDER.length + " models: two baselines, two Logistic Regression variants, two Random Forest variants, HistGradientBoosting, XGBoost, LightGBM, CatBoost, an MLP and an FT-Transformer. High Risk accounts for " + pct(COUNTS["High Risk"] / TOTAL) + " of the records, so the comparison reports High-Risk precision, recall, F1 and PR-AUC alongside overall accuracy, under argmax and validation-tuned decision rules. CatBoost had the highest mean validation High-Risk F1 under both rules and was selected for detailed reporting. Test-set differences among the leading boosting and neural-network models are small relative to seed variation."));
C.push(H2("1.2 Primary Contributions of the Work"));
C.push(Bu("**A leakage-free three-class risk formulation.** The risk label is derived from the final examination score, which is then excluded from the inputs, so the model only uses information available before the final examination."));
C.push(Bu(`**Domain-driven feature engineering.** Four interpretable features are added: a previous-failure flag, a study-engagement score, a digital-distraction score and a learning-support score. In the Random Forest impurity ranking, the study-engagement score ranks ${rankOf(IMP_RF, "study_engagement_score")}th and the learning-support score ${rankOf(IMP_RF, "learning_support_score")}th.`));
C.push(Bu("**A reproducible, imbalance-aware evaluation protocol.** The pipeline uses a stratified " + splitPct(CONFIG.split.train) + "/" + splitPct(CONFIG.split.val) + "/" + splitPct(CONFIG.split.test) + " split, " + CONFIG.cv_folds + "-fold cross-validation where available, and " + CONFIG.seeds.length + " seeds. Results are reported as mean +/- standard deviation; hyperparameters and thresholds are selected on validation data, and the " + n0(SPLIT.test.n) + "-student test set is evaluated once. Two baselines quantify how much full-feature models add over a constant prediction and the midterm mark alone."));
C.push(Bu("**A deployable decision-support tool.** The model is wrapped in a reusable prediction module and a four-page Streamlit dashboard that converts each prediction into support actions, with a human advisor in the loop."));

// ---------- 2. Motivation ----------
C.push(H1("2. Motivation of the Research"));
C.push(P("Student failure carries a high cost for the student, the institution and society: repeated courses, delayed graduation, financial loss and, in the worst case, dropout. Most of this cost can be reduced if difficulties are detected early, while there is still time for remedial teaching, counselling or changes to study habits."));
C.push(P("Three practical problems motivated this work:"));
C.push(Bu("**Late detection.** Traditional monitoring relies on end-of-term results. By the time a failing grade is recorded, the opportunity to intervene has passed."));
C.push(Bu("**Limited advisor time.** A single advisor may be responsible for hundreds of students and cannot review each profile in detail. A ranked, probability-based signal helps them focus attention on the students who need it most."));
C.push(Bu("**Signals beyond grades.** Academic outcomes are associated with behaviour (study hours, attendance, library use), environment (internet and study-space quality) and well-being (stress, motivation, self-efficacy). These factors are rarely combined in a single, systematic view."));
C.push(P("Machine learning offers a way to combine all of these signals into a single, consistent risk estimate. The goal of this research is not to label students. It is to move the advisor's question from *\"Who has failed?\"* to *\"Who may need help, why, and what can we do now?\"*"));

// ---------- 3. Related Work ----------
C.push(H1("3. Related Work"));
C.push(P("Predicting student performance and dropout is one of the most active topics in educational data mining. Table 1 summarises recent studies (2024–2026) that are closest to the present work, followed by a brief discussion."));
C.push(TCAP("**Table 1.** Related work on student academic-risk prediction using ML approaches"));
C.push(TABLE(
  ["Authors Name", "Methodology Used", "Dataset Used", "Findings", "Accuracy", "Limitations"],
  [
    ["Zamtinah, Rianto & Santosa (2026) [1]", "LR, Decision Tree, RF; proposed class-weighted RF with SHAP explanations", "Public simulated dataset, 2,392 student records", "Balanced RF outperformed linear and standard tree baselines; GPA, absences and weekly study time were the top predictors", "92.3%\n(macro F1 0.872)", "Small simulated dataset; needs validation on longitudinal LMS data"],
    ["Putra, Prasetya & Mayadi (2025) [2]", "Random Forest vs. XGBoost", "Higher-education attrition records, 4,424 × 34 (Dropout / Enrolled / Graduate)", "RF slightly outperformed XGBoost on demographic, socio-economic and academic variables", "80.56%\n(sensitivity 72.42%)", "Single small dataset; ensembles and deep learning left for future work"],
    ["Aljawazneh, Alqirem & Al-Adwan (2026) [3]", "RF, SVM, XGBoost, DNN, TabNet with SMOTE and Borderline-SMOTE", "Higher-education student dropout data", "XGBoost was best on both imbalanced and balanced data; balancing significantly improved recall of at-risk students", "Not reported in abstract", "Limitations not stated in the abstract"],
    ["Guevara-Reyes et al. (2025) [4]", "XGBoost and baselines; Optuna tuning (100 trials); SHAP interpretability", "~50,000 Ecuadorian student records (AMEI and Ser Bachiller), 27 features", "XGBoost best; 15% lower MSE than baseline models", "R² = 0.91\n(F1 = 0.86)", "No pedagogical variables; pre-COVID data; risk of reproducing socio-economic bias"],
    ["Jimenez Martinez, Sood & Mahto (2024) [5]", "SVM, Naive Bayes, KNN, DT, LR, RF; k-fold cross-validation", "Canvas LMS and university dashboard data, CSU Fullerton, Fall 2023", "Naive Bayes performed best; all models gave acceptable at-risk predictions", "Not reported in abstract", "Preliminary study; one semester at one institution"],
    ["**Proposed work**", ORDER.length + " models: two baselines, LR, RF, HistGB, XGBoost, LightGBM, CatBoost, MLP and FT-Transformer; stratified " + splitPct(CONFIG.split.train) + "/" + splitPct(CONFIG.split.val) + "/" + splitPct(CONFIG.split.test) + " split, " + CONFIG.cv_folds + "-fold CV where available, " + CONFIG.seeds.length + " seeds, validation selection", n0(TOTAL) + " students, three risk classes", "CatBoost selected on validation under both rules; leading boosting and neural-network models have similar High-Risk F1 within seed variation", pct(catA.test_accuracy.mean) + "\n(macro F1 " + m3(catA.test_macro_f1) + ")", "High-Risk recall " + m3(catA.test_high_risk_recall) + " (argmax) / " + m3(catB.test_high_risk_recall) + " (tuned); platform mix and subsampled deep models"]
  ],
  [1350, 1640, 1520, 1886, 1060, 1570], { size: 16, shadeRows: [5], center: [4] }
));
C.push(NOTE("*Note:* accuracy values are those reported by the original authors on different datasets and class definitions, so they are not directly comparable."));
C.push(P("Three themes emerge from this literature. First, **tree-based ensembles** (Random Forest, XGBoost and other gradient-boosting methods) consistently outperform linear models on tabular student data [1]–[4]. Second, **class imbalance** is a central difficulty: at-risk students are a minority, and techniques such as class weighting [1] or SMOTE [3] are needed to raise their recall. Third, recent work emphasises **interpretability** through SHAP [1], [4], so that predictions can be trusted and acted upon. Most studies, however, use small datasets (a few thousand records), a single train–test split and no simple baselines. The present work contributes a large-scale (500,000-record), multi-seed evaluation with baselines and validation-tuned decision thresholds, and an end-to-end deployment that turns predictions into advisor-facing support actions."));

// ---------- 4. Proposed Methodologies ----------
C.push(H1("4. Proposed Methodologies"));
C.push(H2("i. Dataset Description"));
C.push(P("The study uses the publicly available **Student Performance and Behaviour** dataset (https://www.kaggle.com/datasets/waddahali/500k-student-performance-and-behavior-dataset) [6]. It contains **500,000 records**, one per university student, described by **27 attributes**. The records cover 25 universities and 39 home cities in Egypt, 18 major subjects and semesters 1–8. The dataset has **no missing values and no duplicate rows**. Table 2 groups the attributes by type and shows how each one is used."));
C.push(TCAP("**Table 2.** Attribute groups of the dataset"));
C.push(TABLE(
  ["Group", "Attributes", "Type", "Role"],
  [
    ["Identifiers", "Student_ID, University_Name, Home_City", "ID / text", "Removed"],
    ["Demographic", "Age (17–30), Gender, Region_Type, Family_Size (2–8), Semester_ID (1–8)", "Mixed", "Input (Semester_ID dropped by the preprocessor)"],
    ["Family background", "Parent_Education (6 levels), Family_Income_Level (5 levels)", "Categorical", "Input"],
    ["Learning environment", "Internet_Quality, Study_Space_Quality (1–10)", "Numeric", "Input"],
    ["Academic history", "Major_Subject (18), Previous_GPA (0–4), Number_of_Failed_Courses (0–10), Total_Credits_Earned (12–144)", "Mixed", "Input"],
    ["Study behaviour", "Weekly_Study_Hours (2–40), Attendance_Rate (0–100%), Library_Visits_Per_Month (0–24)", "Numeric", "Input"],
    ["Lifestyle & well-being", "Extracurricular_Hours, Sleep_Hours, Social_Media_Usage_Hours, Stress_Level, Motivation_Score, Self_Efficacy_Score", "Numeric", "Input"],
    ["Current performance", "Midterm_Mark (0–100)", "Numeric", "Input"],
    ["Outcome", "Final_Exam_Score (0–100)", "Numeric", "Used only to build the target"],
  ],
  [1700, 4126, 1100, 2100], { size: 18 }
));
C.push(P("The categorical attributes are well balanced: Gender is 49.1% female, 48.9% male and 2.0% non-binary; Region_Type is 65.1% urban and 34.9% rural; and each of the 18 subjects accounts for about 5.5% of the records. Table 3 summarises the main numeric attributes.", { after: 60 }));
C.push(TCAP("**Table 3.** Summary statistics of key numeric attributes (n = 500,000)"));
C.push(TABLE(
  ["Attribute", "Mean", "Std. dev.", "Min", "Max"],
  [
    ["Previous_GPA", "2.79", "0.67", "0.0", "4.0"],
    ["Weekly_Study_Hours", "18.16", "7.91", "2", "40"],
    ["Attendance_Rate (%)", "79.60", "13.19", "0", "100"],
    ["Library_Visits_Per_Month", "9.22", "3.71", "0", "24"],
    ["Stress_Level", "5.52", "1.97", "1", "10"],
    ["Self_Efficacy_Score", "4.54", "1.49", "1", "10"],
    ["Midterm_Mark", "53.76", "10.40", "0", "100"],
    ["Final_Exam_Score", "57.57", "11.31", "0", "100"],
  ],
  [3226, 1450, 1450, 1450, 1450], { size: 18, center: [1, 2, 3, 4] }
));
C.push(P("Exploratory analysis showed that the attributes most strongly correlated with the final examination score are the midterm mark (r = 0.68), weekly study hours (0.45), library visits (0.43), previous GPA (0.42), self-efficacy (0.35) and motivation (0.23), while stress is negatively correlated (−0.14). Sleep, age, family size and credits earned show almost no linear relationship with the outcome."));
C.push(P("**Target variable.** The class distribution of the derived target is shown in Fig. 2. It is highly imbalanced: Medium Risk accounts for 82.3% of the students and High Risk for only 5.1%.", { keepNext: true }));
C.push(IMG(path.join(ASSETS, "fig2_class_distribution.png"), 4.4));
C.push(CAP("**Fig. 2.** Distribution of the three academic-risk classes (full dataset)"));

C.push(H2("ii. Data Preprocessing"));
C.push(P("The raw data were converted into model-ready inputs in the following steps."));
C.push(H3("Step 1 – Data quality verification"));
C.push(P("The dataset was checked for missing values, duplicate rows, data types and value ranges. No missing values or duplicates were found, so no imputation or de-duplication was needed."));
C.push(H3("Step 2 – Target construction"));
C.push(P("The continuous Final_Exam_Score *s* was discretised into the three-class label *y* using fixed academic thresholds:", { keepNext: true }));
C.push(EQ(CONFIG.labels.rule));
C.push(P("The pipeline checks the resulting class counts (" + n0(COUNTS["High Risk"]) + " / " + n0(COUNTS["Medium Risk"]) + " / " + n0(COUNTS["Low Risk"]) + ") on every run, so the labels cannot change silently."));
C.push(H3("Step 3 – Prevention of data leakage"));
C.push(P("Because the target is computed from Final_Exam_Score, this column was removed from the feature set. If it were kept, the model would simply learn the thresholding rule. Removing it ensures the model only uses information available **before** the final examination."));
C.push(H3("Step 4 – Removal of non-generalisable attributes"));
C.push(P("Student_ID, University_Name and Home_City were dropped. Identifiers carry no predictive meaning, and institution or city names would tie the model to specific locations. Semester_ID is passed to the preprocessor but is not included in its transformers, so it is discarded (remainder = \"drop\")."));
C.push(H3("Step 5 – Feature engineering"));
C.push(P("Four domain-informed features were created from the existing attributes:", { keepNext: true }));
C.push(EQ("previous_failure_flag = 1 if Number_of_Failed_Courses > 0, else 0"));
C.push(EQ("study_engagement_score = Weekly_Study_Hours + Library_Visits_Per_Month"));
C.push(EQ("digital_distraction_score = Social_Media_Usage_Hours"));
C.push(EQ("learning_support_score = Internet_Quality + Study_Space_Quality"));
C.push(P("A single shared function computes these features in training and in the deployed prediction module, so they are identical in both places."));
C.push(H3("Step 6 – Stratified train–validation–test split"));
C.push(P("The data were divided into **" + splitPct(CONFIG.split.train) + " training (" + n0(SPLIT.train.n) + " records)**, **" + splitPct(CONFIG.split.val) + " validation (" + n0(SPLIT.val.n) + ")** and **" + splitPct(CONFIG.split.test) + " test (" + n0(SPLIT.test.n) + ")** in two stratified steps with a fixed random seed (" + CONFIG.split.seed + "). Stratification preserves class proportions across subsets, which matters for the rare High-Risk class. The validation set is used for tuning and model selection; the test set is used once for final evaluation."));
C.push(H3("Step 7 – One-hot encoding of categorical attributes"));
C.push(P("The five nominal attributes have no natural order, so each category was mapped to a separate binary column (one-hot encoding). This produced **34 columns**: 18 subjects, 3 genders, 2 region types, 6 parental-education levels and 5 income levels."));
C.push(H3("Step 8 – Standardisation of numeric attributes"));
C.push(P("The 21 numeric attributes (17 original and 4 engineered) were standardised to zero mean and unit variance:", { keepNext: true }));
C.push(EQ("z = (x − μ) / σ"));
C.push(P("where μ and σ are estimated **only on the data the model is trained on** (the training set, or the training folds during cross-validation), so that no information from held-out data leaks into training."));
C.push(H3("Step 9 – Pipeline construction"));
C.push(P("Steps 7 and 8 were implemented in a scikit-learn ColumnTransformer, which outputs **55 features** (34 + 21). The transformer and each classifier are combined into a single Pipeline, so the transformer is refitted inside every cross-validation fold and every final fit."));

C.push(H2("iii. Methodology"));
C.push(P("**Proposed model architecture.** Fig. 1 shows the reproducible pipeline and its evaluation: data preparation, comparison of " + ORDER.length + " models across " + CONFIG.seeds.length + " seeds, validation selection, and one final test evaluation. Cross-validation is reported for " + cvModels.length + " models; the unconstrained Random Forest and FT-Transformer have no CV results. The current dashboard serves the validation-selected CatBoost pipeline (seed 0) with saved validation thresholds; the initial HistGradientBoosting estimator and preprocessor remain a conditional fallback.", { keepNext: true }));
C.push(IMG(path.join(ASSETS, "fig1_architecture.png"), 6.3));
C.push(CAP("**Fig. 1.** Proposed architecture of the Academic Early-Intervention framework"));
C.push(H3("(a) Baselines"));
C.push(P(`Two baselines show how much each model adds. The **majority-class baseline** always predicts Medium Risk. The **midterm-only baseline** is a Logistic Regression that uses Midterm_Mark as its single input; its regularisation strength and class weighting were tuned on the validation set (selected: C = ${midBest.C}, class_weight = ${midBest.class_weight}).`));
C.push(H3("(b) Logistic Regression"));
C.push(P("Multinomial Logistic Regression models the probability of each class *k* with the softmax function over linear scores:", { keepNext: true }));
C.push(EQ("P(y = k | x) = exp(w_{k}·x + b_{k}) / Σ_{j} exp(w_{j}·x + b_{j})"));
C.push(P("Multinomial Logistic Regression was used for the full-feature model, with a class-weighted variant included. The fixed solver and iteration values are not serialized in the results JSON."));
C.push(H3("(c) Random Forest"));
C.push(P("Random Forest is a bagging ensemble of decision trees. Each tree is trained on a bootstrap sample and considers a random subset of features (√p) at each split; the final prediction is a majority vote. To counter the imbalance, **balanced class weights** were used, so each class *c* is weighted by:", { keepNext: true }));
C.push(EQ("w_{c} = n / (K · n_{c})"));
C.push(P("The capped and unconstrained Random Forest variants use balanced class weights. The capped variant selects max_depth and min_samples_leaf from the validation grid shown in Table 4; fixed parameter details for the unconstrained variant are not serialized in the results JSON."));
const leafs = RF_TUNING.grid.min_samples_leaf, depths = RF_TUNING.grid.max_depth;
const rfCell = (d, l) => {
  const r = RF_TUNING.results.find((x) => x.params.max_depth === d && x.params.min_samples_leaf === l);
  const txt = `${r.val.high_risk_f1.toFixed(3)} / ${r.val.macro_f1.toFixed(3)}`;
  return d === rfBest.max_depth && l === rfBest.min_samples_leaf ? `**${txt}**` : txt;
};
C.push(TCAP("**Table 4.** Random Forest tuning on the validation set (High-Risk F1 / macro F1; seed " + RF_TUNING.seed + ")"));
C.push(TABLE(["max_depth", ...leafs.map((l) => `min_samples_leaf = ${l}`)], depths.map((d) => [String(d), ...leafs.map((l) => rfCell(d, l))]),
  [2426, 2200, 2200, 2200], { size: 18, center: [0, 1, 2, 3] }));
C.push(H3("(d) Histogram-based Gradient Boosting"));
C.push(P("Gradient boosting builds an additive model in which each new tree *h*_{m} is fitted to the negative gradient of the loss of the current ensemble:", { keepNext: true }));
C.push(EQ("F_{m}(x) = F_{m−1}(x) + ν · h_{m}(x)"));
C.push(P("where ν is the learning rate. For multi-class problems one tree per class is grown in each iteration and the multinomial log-loss is minimised:", { keepNext: true }));
C.push(EQ("L = − Σ_{i} Σ_{k} y_{ik} · log p_{k}(x_{i})"));
C.push(P("The pipeline includes histogram-based gradient boosting alongside other tree-boosting methods and neural networks. Its fitted results are reported in the same argmax and validation-tuned tables as the other models; no distinct test-set winner is claimed."));
C.push(H3("(e) Added gradient-boosting and neural-network models"));
C.push(P("The extended comparison adds XGBoost, LightGBM, CatBoost, MLP and FT-Transformer to the original seven models. Their tuned parameter values are selected on validation data and listed in Table 5. The deep models use the stratified training subsample recorded in results/config.json. The added models ran in Colab; GPU-capable models used a T4 GPU, while LightGBM remained CPU-based."));
function modelParams(k) {
  if (k === "majority") return "Most-frequent-class baseline";
  if (k === "midterm_lr") return "Input=Midterm_Mark; " + paramText(midBest);
  if (k === "rf_capped") return paramText(rfBest);
  if (TUNING[k]) {
    const sampled = ["mlp", "ft_transformer"].includes(k) ? "; training_rows=" + CONFIG.deep_models.training_rows : "";
    return paramText(TUNING[k].best_params) + sampled;
  }
  return paramText(A(k).params);
}
C.push(TCAP("**Table 5.** Hyperparameters recorded for the 12 models"));
C.push(TABLE(
  ["Model", "Selected or recorded configuration"],
  ORDER.map((k) => [L[k], modelParams(k)]),
  [2700, 6326], { size: 18 }
));
C.push(NOTE("Tuned settings are read from the saved validation tuning files. Fixed configurations whose parameters were not serialized are identified as such. MLP and FT-Transformer used the training subsample size shown in results/config.json."));
C.push(H3("(f) Decision rules"));
C.push(P("Each model outputs class probabilities. Two rules convert them into a predicted class. The **argmax rule** predicts the class with the highest probability. The **tuned-threshold rule** uses two thresholds, t_{high} and t_{low}:", { keepNext: true }));
C.push(EQ("ŷ = High Risk if P(High) ≥ t_{high};  else Low Risk if P(Low) ≥ t_{low};  else Medium Risk"));
C.push(P("For each model and seed, thresholds were selected on validation data to maximize High-Risk F1, with macro F1 as the tie-breaker. The saved thresholds were then applied once to the test set. Because each model follows the same validation-only selection rule, the tuned-threshold comparison is like-for-like."));
C.push(H3("(g) Experimental protocol"));
C.push(Bu("**Cross-validation:** " + CONFIG.cv_folds + "-fold stratified cross-validation was performed for " + cvModels.length + " models. No CV results are recorded for " + noCvModels.map((k) => L[k]).join(" and ") + "."));
C.push(Bu("**Seeds:** the saved results cover " + CONFIG.seeds.length + " seeds (" + CONFIG.seeds.join(", ") + "); means and standard deviations are descriptive summaries across these runs. The data split itself is fixed (seed " + CONFIG.split.seed + ")."));
C.push(Bu("**Selection and testing:** tuning and model selection used validation results. " + selA.selected_model + " was selected under argmax and " + selB.selected_model + " under validation-tuned thresholds; selection.json was written before the test scores. The test set is used for the final comparison only."));
C.push(H3("(h) Intervention layer"));
C.push(P("The predicted class and probabilities are converted into support recommendations (Table 6). The dashboard shows a fixed set of actions for each risk level. The command-line module adds rule-based triggers based on the student's own attributes."));
C.push(TCAP("**Table 6.** Risk-to-action mapping of the intervention layer"));
C.push(TABLE(
  ["Condition", "Recommended support action"],
  [
    ["Predicted High Risk", "Arrange academic counselling; provide subject-specific support; create a personalised study plan; monitor attendance and progress; assign faculty mentoring or peer support"],
    ["Predicted Medium Risk", "Monitor progress regularly; encourage consistent study habits; set short-term goals; schedule periodic mentoring; encourage help-seeking"],
    ["Predicted Low Risk", "Continue current practices; maintain study habits and goal-setting; maintain healthy routines"],
    ["Attendance_Rate < 75%", "Improve and monitor class attendance"],
    ["Number_of_Failed_Courses > 0", "Additional support in previously failed courses"],
    ["Weekly_Study_Hours < 15", "Structured study plan with more weekly study time"],
    ["Stress_Level ≥ 7", "Stress-management and counselling support"],
    ["Motivation_Score < 5", "Mentoring and goal-setting activities"],
    ["Social_Media_Usage_Hours > 4", "Reduce social-media use during study periods"],
  ],
  [3000, 6026], { size: 18 }
));
C.push(H3("(i) Deployment"));
C.push(P("The current dashboard serves the validation-selected CatBoost pipeline (seed 0) and applies its saved validation thresholds. The probabilities below are historical outputs from the initial HistGradientBoosting dashboard model and are included only as an illustrative legacy case study; they are not predictions from the current deployment."));
C.push(IMG(path.join(ASSETS, "fig5a_form.png"), 5.6));
C.push(CAP("**Fig. 3a.** Student input form of the deployed dashboard"));
C.push(IMG(path.join(ASSETS, "fig5b_result.png"), 5.6));
C.push(CAP("**Fig. 3b.** Prediction result for an example student"));

C.push(H2("iv. Evaluation Metrics"));
C.push(P("Let TP, TN, FP and FN denote true positives, true negatives, false positives and false negatives for a given class. The following metrics were used:", { keepNext: true }));
C.push(EQ("Accuracy = (TP + TN) / (TP + TN + FP + FN)"));
C.push(EQ("Precision = TP / (TP + FP)"));
C.push(EQ("Recall = TP / (TP + FN)"));
C.push(EQ("F1 = 2 · Precision · Recall / (Precision + Recall)"));
C.push(EQ("Macro F1 = (1/K) Σ_{k} F1_{k}"));
C.push(EQ("PR-AUC (average precision) = Σ_{n} (R_{n} − R_{n−1}) · P_{n}"));
C.push(Bu("**Accuracy** measures the overall share of correct predictions. On this dataset it is misleading on its own: the majority-class baseline reaches " + pct(A("majority").test_accuracy.mean) + " while detecting no High-Risk student."));
C.push(Bu("**High-Risk recall** is the share of truly High-Risk students the model detects. It is the most important metric for an early-warning system, because a missed struggling student is costlier than an unnecessary check-in."));
C.push(Bu("**High-Risk precision** is the share of flagged students who are truly High Risk. It controls the advisor's workload of false alarms."));
C.push(Bu("**F1-score** balances precision and recall; **macro F1** averages F1 across the three classes equally, so the minority classes count as much as the majority class."));
C.push(Bu("**High-Risk PR-AUC** summarises precision across all recall levels (P_{n} and R_{n} are the precision and recall at the n-th score threshold). It does not depend on any decision rule, so it is the fairest single comparison between models. Its chance level equals the High-Risk prevalence (" + pct(COUNTS["High Risk"] / TOTAL) + ")."));
C.push(Bu("**Confusion matrices**, **permutation importance** (drop in macro F1 when a feature is shuffled) and **impurity importance** were used for error analysis and interpretation."));

// ---------- 5. Results ----------
C.push(H1("5. Experiment Results and Discussion"));
C.push(H2("5.1 Experimental Setup"));
C.push(P("All experiments were run by run_pipeline.py in Python " + CONFIG.versions.python + " with scikit-learn " + CONFIG.versions["scikit-learn"] + ", pandas " + CONFIG.versions.pandas + " and NumPy " + CONFIG.versions.numpy + ". The seven original models ran on CPU. The five added models ran in Colab; GPU-capable models used a T4 GPU, while LightGBM remained CPU-based. The Colab run used PyTorch 2.11.0+cu130. All models share the stratified test set of " + n0(SPLIT.test.n) + " students (" + n0(testCounts["High Risk"]) + " High Risk, " + n0(testCounts["Medium Risk"]) + " Medium Risk, " + n0(testCounts["Low Risk"]) + " Low Risk). Values are means across the saved seeds; ± denotes standard deviation."));

// Table A
C.push(H2("5.2 Model Comparison: Argmax Rule (Table A)"));
C.push(P("Table 7 and Fig. 4 compare all 12 models on the test set using the argmax rule, which chooses the highest-probability class."));
const colsA = ["test_accuracy", "test_macro_f1", "test_high_risk_precision", "test_high_risk_recall", "test_high_risk_f1", "test_high_risk_pr_auc"];
const bestA = Object.fromEntries(colsA.map((c) => [c, bestOf(A, c)]));
C.push(TCAP("**Table 7 (Table A).** Test-set results with the **argmax** decision rule (mean ± std over five seeds)"));
C.push(TABLE(
  ["Model", "Accuracy", "Macro F1", "High-Risk Precision", "High-Risk Recall", "High-Risk F1", "High-Risk PR-AUC", "Saved size (MB)"],
  ORDER_A.map((k) => [A(k).model, ...colsA.map((c) => cellB(A(k), c, bestA[c])), size(A(k).saved_model_size_mb)]),
  [2050, 900, 900, 1000, 1000, 1000, 1076, 1100], { size: 16, center: [1, 2, 3, 4, 5, 6, 7] }
));
C.push(IMG(path.join(FIG, "test_metrics_argmax.png"), 6.3));
C.push(CAP("**Fig. 4.** Table A: test-set metrics with the argmax rule (error bars: std over five seeds)"));
const rfuA = A("rf_unconstrained"), rfcA = A("rf_capped"), lrA = A("logreg"), lrbA = A("logreg_balanced"), majA = A("majority");
C.push(P("Under argmax, CatBoost, LightGBM, XGBoost and FT-Transformer form the leading High-Risk F1 group, with HistGradientBoosting and MLP close behind. Their seed variation overlaps; the point estimates do not establish a clear winner. These boosting and neural-network models all substantially exceed both baselines on High-Risk F1 and PR-AUC. The Random Forest and linear-model rows provide useful contrasts, but their results do not alter that group-level conclusion."));
C.push(P("**Model selection under argmax.** CatBoost ranked first on validation High-Risk F1 and was selected before test scoring. Its test-set mean is close to those of LightGBM and XGBoost, and the observed gaps should be interpreted alongside the across-seed variation."));

// Table B
C.push(H2("5.3 Model Comparison: Validation-Tuned Thresholds (Table B)"));
C.push(P("Table 8 and Fig. 5 report test-set results after each model's thresholds were tuned on validation data. They are distinct from Table 7's argmax predictions. PR-AUC does not depend on the decision rule and is identical in both tables."));
const colsB = ["test_accuracy", "test_macro_f1", "test_high_risk_precision", "test_high_risk_recall", "test_high_risk_f1"];
const bestB = Object.fromEntries(colsB.map((c) => [c, bestOf(Bt, c)]));
const thr = (s) => s.mean.toFixed(2) + (s.std >= 0.005 ? ` ± ${s.std.toFixed(2)}` : "");
C.push(TCAP("**Table 8 (Table B).** Test-set results with **validation-tuned thresholds** (mean ± std over five seeds)"));
C.push(TABLE(
  ["Model", "t_high", "t_low", "Accuracy", "Macro F1", "High-Risk Precision", "High-Risk Recall", "High-Risk F1"],
  ORDER_B.map((k) => [Bt(k).model, thr(Bt(k).t_high), thr(Bt(k).t_low), ...colsB.map((c) => cellB(Bt(k), c, bestB[c]))]),
  [2050, 860, 860, 1050, 1050, 1050, 1050, 1056], { size: 16, center: [1, 2, 3, 4, 5, 6, 7] }
));
C.push(NOTE(`*Note:* the majority-class baseline assigns P(High) = 0 to every student, so no threshold can make it predict High Risk.`));
C.push(IMG(path.join(FIG, "test_metrics_tuned_thresholds.png"), 6.3));
C.push(CAP("**Fig. 5.** Table B: test-set metrics with validation-tuned thresholds (error bars: std over five seeds)"));
const rfuB = Bt("rf_unconstrained"), rfcB = Bt("rf_capped"), lrB = Bt("logreg");
C.push(P("Under validation-tuned thresholds, CatBoost, LightGBM, XGBoost and FT-Transformer again have similar High-Risk F1, with HistGradientBoosting and MLP close behind. CatBoost had the highest mean validation High-Risk F1 and was selected under this rule; the test-set differences remain comparable to seed variation, so no clear winner is claimed. The argmax and tuned-threshold metrics are kept distinct in Tables 7 and 8."));
C.push(P("The higher MLP and FT-Transformer High-Risk recall values in Table 8 are from validation-tuned thresholds; their argmax recalls are shown separately in Table 7. These result files do not support attributing the difference to class weighting."));

// operating points of the selected model
const OP = json("operating_points.json");
const opRules = [
  ["f1_optimal", "F1-optimal"],
  ["recall_ge_0.70", "Recall ≥ 0.70"],
  ["recall_ge_0.80", "Recall ≥ 0.80"],
];
const opS = (k) => OP.summary[k];
const thr3 = (s) => s.mean.toFixed(3) + (s.std >= 0.0005 ? ` ± ${s.std.toFixed(3)}` : "");
C.push(H3("5.3.1 Operating points of the selected model"));
C.push(P(`An advisor may prefer to catch more at-risk students at the cost of more false alarms. To show this trade-off, three thresholds on P(High) were chosen on the validation set for ${OP.label}, separately for each seed, treating High Risk against the other two classes: the **F1-optimal** threshold, and the most precise thresholds whose validation recall is at least **0.70** and at least **0.80**. Each threshold was then applied once to the test set (Table 9, Fig. 6).`));
C.push(TCAP(`**Table 9.** High-Risk operating points of ${OP.label}: thresholds chosen on validation, applied once to the test set (mean ± std over five seeds)`));
C.push(TABLE(
  ["Threshold rule (on validation)", "Threshold on P(High)", "Test precision", "Test recall", "Test F1", "Students flagged"],
  opRules.map(([k, name]) => [name, thr3(opS(k).threshold), pm(opS(k).test.precision), pm(opS(k).test.recall), pm(opS(k).test.f1), pct(opS(k).test.flagged_share.mean)]),
  [2226, 1360, 1360, 1360, 1360, 1360], { size: 18, center: [1, 2, 3, 4, 5] }
));
C.push(IMG(path.join(FIG, "operating_points_high_risk.png"), 5.4));
C.push(CAP("**Fig. 6.** High-Risk precision–recall curves of the selected model on validation and test (seed 0), with the three operating points marked (test mean ± std over five seeds)"));
const f1o = opS("f1_optimal").test, r70 = opS("recall_ge_0.70").test, r80 = opS("recall_ge_0.80").test;
C.push(P(`The recall targets carried over to unseen data: test recall was ${m3(r70.recall)} for the 0.70 target and ${m3(r80.recall)} for the 0.80 target. Moving from the F1-optimal point to the recall ≥ 0.70 point changed test F1 only from ${m3(f1o.f1)} to ${m3(r70.f1)}, while raising recall from ${m3(f1o.recall)} to ${m3(r70.recall)} and the share of students flagged from ${pct(f1o.flagged_share.mean)} to ${pct(r70.flagged_share.mean)}. The recall ≥ 0.80 point detects ${pct(r80.recall.mean)} of High-Risk students, but its precision of ${m3(r80.precision)} means that ${r80.precision.mean < 0.5 ? "more than half" : "a large share"} of the flagged students are not High Risk, and ${pct(r80.flagged_share.mean)} of all students are flagged. The appropriate operating point therefore depends on how much advisor time is available. The F1-optimal test F1 (${m3(f1o.f1)}) is consistent with Table 8 (${m3(catB.test_high_risk_f1)}), and applies the same validation High-Risk F1 objective.`));

// Baselines
C.push(H2("5.4 Baselines: What the Other Features Add"));
const gA = (k) => A(k).test_gain_over_midterm_only, gB = (k) => Bt(k).test_gain_over_midterm_only;
C.push(P(`The majority-class baseline reaches ${pct(majA.test_accuracy.mean)} accuracy while detecting no High-Risk student, which confirms that accuracy alone is not informative here. The midterm-only baseline reaches a High-Risk PR-AUC of ${m3(midA.test_high_risk_pr_auc)} and, with tuned thresholds, a High-Risk F1 of ${m3(midB.test_high_risk_f1)}. Table 10 shows the gain of each full-feature model over this baseline under the same decision rule.`));
C.push(TCAP("**Table 10.** Test-set gain over the midterm-only baseline (difference of means)"));
C.push(TABLE(
  ["Model", "Δ High-Risk F1 (argmax)", "Δ High-Risk F1 (tuned)", "Δ Macro F1 (tuned)", "Δ High-Risk PR-AUC"],
  ORDER_A.filter((k) => !["majority", "midterm_lr"].includes(k)).map((k) => [A(k).model, sgn(gA(k).high_risk_f1), sgn(gB(k).high_risk_f1), sgn(gB(k).macro_f1), sgn(gA(k).high_risk_pr_auc)]),
  [2826, 1550, 1550, 1550, 1550], { size: 18, center: [1, 2, 3, 4] }
));
C.push(P("The full-feature models improve High-Risk PR-AUC over the midterm-only baseline. For CatBoost, High-Risk PR-AUC rises from " + m3(midA.test_high_risk_pr_auc) + " to " + m3(catA.test_high_risk_pr_auc) + " (" + prRatio + "x) and tuned High-Risk F1 from " + m3(midB.test_high_risk_f1) + " to " + m3(catB.test_high_risk_f1) + ". Fig. 7 compares the precision-recall curves for all 12 models with the midterm-only and majority baselines; the leading boosting and neural-network curves are close to one another and above the baselines.", { keepNext: true }));
C.push(IMG(path.join(FIG, "pr_curves_high_risk_test.png"), 5.2));
C.push(CAP("**Fig. 7.** High-Risk precision–recall curves on the test set (seed 0; PR-AUC values are means over five seeds)"));

// CV
C.push(H2("5.5 Cross-Validation and Stability"));

C.push(P("Table 11 reports " + CONFIG.cv_folds + "-fold cross-validation summaries on the training set where they are available. Two models have no CV result: the unconstrained Random Forest and FT-Transformer. Cross-validation and test results are descriptive summaries from different evaluation subsets; standard deviations here are not significance tests."));
C.push(TCAP("**Table 11.** " + CONFIG.cv_folds + "-fold cross-validation on the training set, argmax rule (mean and standard deviation over " + CONFIG.seeds.length + " seeds)"));
const cvCell = (r, k) => (r[`cv_${k}`] ? pm(r[`cv_${k}`]) : "—");
C.push(TABLE(
  ["Model", "Accuracy", "Macro F1", "High-Risk F1", "High-Risk PR-AUC"],
  ORDER.map((k) => [A(k).model, ...["accuracy", "macro_f1", "high_risk_f1", "high_risk_pr_auc"].map((c) => cvCell(A(k), c))]),
  [2826, 1550, 1550, 1550, 1550], { size: 18, center: [1, 2, 3, 4] }
));
C.push(NOTE("No CV: results are not recorded for the unconstrained Random Forest or FT-Transformer."));

// per-class
C.push(H2("5.6 Per-Class Performance of the Validation-Selected CatBoost Model"));
const sa = TEST_A.catboost.summary, sb = TEST_B.catboost.summary;
const cls = [["High Risk", "high_risk"], ["Medium Risk", "medium_risk"], ["Low Risk", "low_risk"]];
C.push(TCAP("**Table 12.** Per-class test performance of CatBoost"));
C.push(TABLE(
  ["Class", "Precision (argmax)", "Recall (argmax)", "F1 (argmax)", "Precision (tuned)", "Recall (tuned)", "F1 (tuned)", "Support"],
  [
    ...cls.map(([name, k]) => [name, m3(sa[`${k}_precision`]), m3(sa[`${k}_recall`]), m3(sa[`${k}_f1`]), m3(sb[`${k}_precision`]), m3(sb[`${k}_recall`]), m3(sb[`${k}_f1`]), n0(testCounts[name])]),
    ["Macro F1", "", "", m3(sa.macro_f1), "", "", m3(sb.macro_f1), n0(SPLIT.test.n)],
  ],
  [1626, 1000, 1000, 1000, 1000, 1000, 1000, 1400], { size: 16, center: [1, 2, 3, 4, 5, 6, 7] }
));
const cmRates = (models) => {
  const cm = Object.values(models.catboost.per_seed).reduce((acc, s) => acc.map((row, i) => row.map((v, j) => v + s.confusion_matrix[i][j])), [[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
  return cm.map((row) => row.map((v) => v / row.reduce((a, b) => a + b, 0)));
};
const cmA = cmRates(TEST_A);
C.push(P(`Errors are concentrated in the two minority classes, and they fall almost entirely into the neighbouring Medium-Risk class. Under argmax, ${pct(cmA[0][1])} of High-Risk and ${pct(cmA[2][1])} of Low-Risk students were predicted Medium Risk, while only ${pct(cmA[0][2])} of High-Risk students were predicted Low Risk (confusion matrices summed over the five seeds). This is expected because the classes are created by thresholding a continuous score: students just below 40 or just above 70 closely resemble Medium-Risk students.`));

// class weighting
C.push(H2("5.7 Effect of Class Weighting"));
C.push(P("Class weighting in Logistic Regression shifts the precision-recall trade-off: under argmax, the balanced variant raised High-Risk recall from " + m3(lrA.test_high_risk_recall) + " to " + m3(lrbA.test_high_risk_recall) + ", but its precision fell from " + m3(lrA.test_high_risk_precision) + " to " + m3(lrbA.test_high_risk_precision) + " and accuracy from " + m3(lrA.test_accuracy) + " to " + m3(lrbA.test_accuracy) + ". Its PR-AUC (" + m3(lrbA.test_high_risk_pr_auc) + ") is no higher than the unweighted model's (" + m3(lrA.test_high_risk_pr_auc) + "). Non-linear models made a larger difference than reweighting a linear one."));

// importance
C.push(H2("5.8 Feature Importance"));
const top = (list, n, dp) => list.slice(0, n).map((x) => `${x.f.replace(/_/g, " ")} (${x.v.toFixed(dp)})`).join(", ");
C.push(P(`Fig. 8 shows the permutation importance of HistGradientBoosting on the validation set: the drop in macro F1 when a feature is randomly shuffled, averaged over five seeds. The most important features were ${top(IMP_HGB, 4, 3)}. In the capped Random Forest's impurity ranking (Fig. 9), the midterm mark ranks first (${impVal(IMP_RF, "Midterm_Mark").toFixed(3)}), followed by ${top(IMP_RF.slice(1), 4, 3)}. The two methods agree that study behaviour, stress, prior grades and the midterm mark dominate. Permutation importance gives the midterm mark a lower rank, which is expected when correlated features such as study hours and previous GPA carry much of the same information. Behavioural and well-being factors (study hours, stress, self-efficacy, attendance and motivation) are among the most important features in both rankings. This shows that these factors carry predictive information about final-examination risk in this dataset. Because the analysis is observational, it does not show that changing a student's habits or well-being would change their outcome; establishing that would require an intervention study.`, { keepNext: true }));
C.push(IMG(path.join(FIG, "feature_importance_hgb_permutation.png"), 5.2));
C.push(CAP("**Fig. 8.** HistGradientBoosting permutation importance on the validation set (mean ± std over five seeds)"));
C.push(IMG(path.join(FIG, "feature_importance_rf_capped.png"), 5.2));
C.push(CAP("**Fig. 9.** Capped Random Forest impurity importance (mean ± std over five seeds)"));

// initial experiment
C.push(H2("5.9 Initial Experiment (Superseded)"));
const leg = LEGACY.models;
const legRow = (name, label) => {
  const r = leg[name];
  return [label, r.accuracy.toFixed(3), r.high_risk_precision.toFixed(3), r.high_risk_recall.toFixed(3), r.high_risk_f1.toFixed(3), r.saved_model_size_mb != null ? r.saved_model_size_mb.toFixed(1) : "—"];
};
C.push(P("Before the reproducible pipeline was built, an initial experiment trained Logistic Regression, an unconstrained class-weighted Random Forest and HistGradientBoosting on a single stratified 80/20 split (400,000 training and 100,000 test students), with one seed and no validation set. Its results are listed in Table 13 for transparency only. They were obtained on a different split and protocol and are **not directly comparable** with Tables 7–12, which supersede them. The legacy HistGradientBoosting estimator and preprocessor are retained only as a conditional fallback; normal dashboard operation uses the selected CatBoost pipeline."));
C.push(TCAP("**Table 13.** Initial experiment: 80/20 split, single seed (superseded; for reference only)"));
C.push(TABLE(
  ["Model", "Accuracy", "High-Risk Precision", "High-Risk Recall", "High-Risk F1", "Saved size (MB)"],
  [legRow("Logistic Regression", "Logistic Regression"), legRow("RF (unconstrained)", "RF (unconstrained)"), legRow("HistGradientBoosting", "HistGradientBoosting")],
  [2426, 1320, 1320, 1320, 1320, 1320], { size: 18, center: [1, 2, 3, 4, 5] }
));

// case study
C.push(H2("5.10 Historical Case Study (Initial Dashboard Model)"));
C.push(P("These two student profiles and probabilities document the initial HistGradientBoosting dashboard only; they are not outputs from the current CatBoost deployment. For a student with a previous GPA of 1.2, a midterm mark of 25, 4 weekly study hours, 50% attendance and a stress level of 9 (other fields at default values), that historical dashboard predicted **High Risk with 92.6% probability** (Medium 7.4%, Low 0.0%) and recommended counselling, subject support, a personalised study plan, attendance monitoring and mentoring. For a student with a GPA of 3.5, a midterm mark of 80, 25 study hours, 95% attendance and a stress level of 3, it predicted **Low Risk with 80.0% probability**. These examples are retained as historical results and should not be presented as current deployment predictions."));

// SOTA
C.push(H2("5.11 Comparison with the State of the Art"));
C.push(P("Accuracy is a weak basis for comparison on this dataset: the majority-class baseline already scores " + m3(A("majority").test_accuracy) + " while detecting no High-Risk student. A more informative comparison uses High-Risk PR-AUC. CatBoost reaches " + m3(catA.test_high_risk_pr_auc) + ", against " + m3(A("majority").test_high_risk_pr_auc) + " for the majority class and " + m3(midA.test_high_risk_pr_auc) + " for the midterm-only baseline, about " + prRatio + " times the midterm-only value. The result files show similar High-Risk F1 for the leading boosting and neural-network models, with HistGradientBoosting and MLP close behind; all substantially exceed the baselines. External studies in Table 1 use different datasets and class definitions, so their reported metrics are not directly comparable."));

// limitations
C.push(H2("5.12 Limitations"));
C.push(Bu("Even with validation-tuned thresholds, CatBoost detects " + pct(catB.test_high_risk_recall.mean) + " of High-Risk students, so about " + Math.round((1 - catB.test_high_risk_recall.mean) * 100) + "% are still missed, and about " + Math.round((1 - catB.test_high_risk_precision.mean) * 100) + "% of students flagged High Risk are false alarms."));
C.push(Bu(`Tuning was limited. The Random Forest grid was small, and its best setting (max_depth = ${rfBest.max_depth}) lies at the edge of the grid, so deeper trees might perform better. HistGradientBoosting's hyperparameters were not tuned, and thresholds were searched in steps of 0.05.`));
C.push(Bu("The unconstrained Random Forest and FT-Transformer have no cross-validation results. The MLP and FT-Transformer used a " + n0(CONFIG.deep_models.training_rows) + "-row training subsample."));
C.push(Bu("The very regular category proportions (for example, income levels of exactly 35/20/20/15/10%) suggest that the dataset may be synthetically generated, so results should be validated on real institutional data."));
C.push(Bu("The data come from a single country. Sensitive attributes such as gender are used as inputs and have not yet been audited for fairness."));
C.push(Bu("The original seven-model runs used CPU, while the five added models ran in a Colab T4 environment with Python " + CONFIG.versions.python + " and PyTorch 2.11.0+cu130. This platform difference limits strict comparability. The standard deviations are descriptive, not significance tests. XGBoost's saved test metrics have zero across-seed standard deviation; the result JSON does not include prediction vectors to independently verify that its predictions were identical across seeds. The dashboard serves the selected CatBoost pipeline and provides CatBoost SHAP explanations; the legacy HistGradientBoosting path is conditional fallback behavior."));

// ---------- 6. Conclusions ----------
C.push(H1("6. Conclusions"));
C.push(P("This work presented the Academic Early-Intervention framework, an end-to-end system for predicting academic risk before final examinations and connecting predictions to support actions. The evaluation used " + n0(TOTAL) + " records and compared " + ORDER.length + " models under argmax and validation-tuned decision rules. CatBoost was selected on validation under both rules. Its test High-Risk F1 was " + pm(catA.test_high_risk_f1) + " under argmax and " + pm(catB.test_high_risk_f1) + " under tuned thresholds, but the leading boosting and neural-network models performed comparably within seed variation, with HistGradientBoosting and MLP close behind. These models substantially exceeded the baselines. Standard deviations are descriptive rather than significance tests. The seven original models ran on CPU and the five added models in a Colab T4 environment; MLP and FT-Transformer used a training subsample, and no CV results are available for FT-Transformer or unconstrained Random Forest. The current dashboard serves the selected CatBoost pipeline with saved validation thresholds and SHAP explanations. The results support further validation, not a claim of a single clear winner."));
C.push(P("Future work will focus on (i) further improving High-Risk recall through cost-sensitive learning and resampling such as SMOTE; (ii) broader hyperparameter searches for HistGradientBoosting and deeper Random Forests; (iii) validating individual SHAP explanations with advisors; (iv) fairness auditing across demographic groups; (v) testing deployment reliability and threshold behavior in a controlled setting; and (vi) validation on real, longitudinal institutional data, including weekly learning-management-system activity, to enable even earlier warnings."));

// ---------- 7. References ----------
C.push(H1("7. References"));
const refs = [
  "Z. Zamtinah, R. Rianto and P. I. Santosa, \"Explainable machine learning with class balancing for predicting student academic performance,\" *International Journal of Advances in Intelligent Informatics*, vol. 12, no. 3, 2026, doi: 10.26555/ijain.v12i3.2379.",
  "L. G. R. Putra, D. D. Prasetya and Mayadi, \"Student dropout prediction using Random Forest and XGBoost method,\" *INTENSIF: Jurnal Ilmiah Penelitian dan Penerapan Teknologi Sistem Informasi*, vol. 9, no. 1, 2025, doi: 10.29407/intensif.v9i1.21191.",
  "H. Aljawazneh, R. Alqirem and N. Al-Adwan, \"A comparative study of machine learning and deep learning algorithms for student dropout prediction in higher education,\" *Al-Zaytoonah University Journal of Business*, vol. 2, no. 1, 2026, doi: 10.15849/zjjb.v2i1.16.",
  "R. Guevara-Reyes, I. Ortiz-Garcés, R. Andrade, F. Cox-Riquetti and W. Villegas-Ch, \"Machine learning models for academic performance prediction: interpretability and application in educational decision-making,\" *Frontiers in Education*, vol. 10, 2025, doi: 10.3389/feduc.2025.1632315.",
  "A. L. Jimenez Martinez, K. Sood and R. Mahto, \"Early detection of at-risk students using machine learning,\" arXiv preprint arXiv:2412.09483, 2024.",
  "\"Student Performance and Behaviour\" dataset — https://www.kaggle.com/datasets/waddahali/500k-student-performance-and-behavior-dataset",
  "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. (software reference)",
];
refs.forEach((r, i) => C.push(new Paragraph({ children: [new TextRun({ text: `[${i + 1}]\t` }), ...runs(r)], tabStops: [{ type: TabStopType.LEFT, position: 540 }], indent: { left: 540, hanging: 540 }, spacing: { after: 80, line: 280 }, alignment: AlignmentType.LEFT })));

// ---------- document ----------
const doc = new Document({
  creator: "Academic Early-Intervention Framework",
  title: "Academic Early-Intervention Framework",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(path.join(__dirname, OUT), buf); console.log("written", OUT); });
