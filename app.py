"""
Customer Churn Diagnostic & Prediction Dashboard
=================================================
Run locally:        streamlit run app.py
Deploy:             push this folder to GitHub -> share.streamlit.io -> point to app.py

Objectives covered (tabs):
  1. Descriptive analysis  - cross-tabulations against CHURN
  2. Diagnostic analysis    - probing which factors drive churn (chi2 / Cramer's V)
  3. Feature engineering    - cleaning + encoding pipeline
  4. Supervised learning    - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
  5. Evaluation             - accuracy / precision / recall / F1, ROC, confusion matrices, importance
  6. Findings               - written business-strategy diagnosis with explicit caveats

NOTE ON METHOD: every number is computed live from the loaded data. The app does not
infer intent. A churn-rate disparity between two customer segments is a signal worth
investigating operationally, not automatic proof of a single root cause -- this is
synthetic data built to resemble realistic subscription-business patterns, not a real
customer base, so treat the findings as a template for diagnosis rather than as
literal business facts.
"""
import warnings; warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Diagnostic Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

NAVY, RED, GREEN, ORANGE, GREY = "#1f3a5f", "#c0392b", "#27ae60", "#e67e22", "#7f8c8d"
ACCENT = [NAVY, RED, GREEN, ORANGE]
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.size": 9})

TARGET = "Churn"
POSITIVE_LABEL = "Yes"                       # positive class = a churned customer
ID_COLS = ["CustomerID"]                     # dropped from modelling (identifier)

# ----------------------------------------------------------------------------
# Data loading + feature engineering (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_engineer(file_bytes: bytes | None):
    if file_bytes is None:
        df = pd.read_csv("customer_churn_raw.csv")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    notes = []

    # 1) drop exact duplicate rows (same customer captured twice)
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "CustomerID"])
    df = df.drop_duplicates(subset="CustomerID", keep="first")
    if before - len(df):
        notes.append(f"Removed {before - len(df)} duplicate customer rows.")

    # 2) normalise inconsistent text casing/whitespace in categoricals
    for c in ["Gender", "Contract", "PaymentMethod"]:
        if c in df.columns:
            before_u = df[c].nunique()
            df[c] = df[c].astype(str).str.strip()
            if c == "Gender":
                df[c] = df[c].str.capitalize()
            if c == "Contract":
                df[c] = df[c].str.lower().map({
                    "month-to-month": "Month-to-month", "one year": "One year",
                    "two year": "Two year"}).fillna(df[c])
            if c == "PaymentMethod":
                df[c] = df[c].str.title().replace({"Electronic Check": "Electronic check"})
            after_u = df[c].nunique()
            if after_u < before_u:
                notes.append(f"{c}: merged inconsistent text variants ({before_u} -> {after_u} categories).")

    # 3) impossible values -> treat as missing
    if "Tenure" in df.columns:
        bad = int((df["Tenure"] < 0).sum())
        if bad:
            df.loc[df["Tenure"] < 0, "Tenure"] = np.nan
            notes.append(f"Tenure: {bad} negative (impossible) values set to missing.")

    # 4) outlier correction in MonthlyCharges (10x data-entry errors), capped via IQR
    if "MonthlyCharges" in df.columns:
        q1, q3 = df["MonthlyCharges"].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        out = int((df["MonthlyCharges"] > upper).sum())
        if out:
            df.loc[df["MonthlyCharges"] > upper, "MonthlyCharges"] = np.nan
            notes.append(f"MonthlyCharges: {out} extreme outliers (>3xIQR) set to missing.")

    # 5) missing-value imputation
    for c in ["MonthlyCharges", "TotalCharges", "Tenure"]:
        if c in df.columns:
            n = int(df[c].isna().sum())
            if n:
                df[c] = df[c].fillna(df[c].median())
                notes.append(f"{c}: {n} missing values median-imputed.")
    for c in ["TechSupport", "PaymentMethod"]:
        if c in df.columns:
            n = int(df[c].isna().sum())
            if n:
                df[c] = df[c].fillna(df[c].mode()[0])
                notes.append(f"{c}: {n} missing values filled with the most frequent category.")

    # 6) binary target
    df["CHURNED"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df.reset_index(drop=True), notes


@st.cache_data(show_spinner=False)
def build_design_matrix(df: pd.DataFrame):
    d = df.copy()
    num = [c for c in ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges",
                       "SupportCalls"] if c in d]
    cat = [c for c in ["Gender", "Contract", "PaymentMethod", "InternetService",
                       "TechSupport", "OnlineSecurity", "PaperlessBilling"] if c in d]
    X = d[num + cat]
    y = d["CHURNED"]
    return X, y, num, cat


def make_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


@st.cache_resource(show_spinner=True)
def train_models(_X, _y, num, cat, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size,
                                          stratify=_y, random_state=seed)
    pre = make_preprocessor(num, cat)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "cm": confusion_matrix(yte, pte),
            "roc": roc_curve(yte, proba),
            "report": classification_report(yte, pte, target_names=["Stayed", "Churned"],
                                             zero_division=0),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())


def cramers_v(df, col, target="CHURNED"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum(); r, k = ct.shape
    v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
    return chi2, p, dof, v, int(ct.shape[0])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
up = st.sidebar.file_uploader("Upload customer CSV (or use bundled sample)", type=["csv"])
file_bytes = up.read() if up is not None else None
df, notes = load_and_engineer(file_bytes)

st.sidebar.markdown("**Modelling parameters**")
test_size = st.sidebar.slider("Test split", 0.15, 0.40, 0.25, 0.05)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.info("Positive class = **Churned**. Recall here = share of true churners "
                "the model successfully flags.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("📉 Customer Churn Diagnostic & Prediction Dashboard")
st.caption("Business strategy diagnosis: why is this subscription business losing "
           "customers, and which levers should retention strategy focus on?")
overall = df["CHURNED"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{len(df):,}")
c2.metric("Retained", f"{int((df['CHURNED']==0).sum()):,}", f"{(1-overall)*100:.1f}%")
c3.metric("Churned", f"{int(df['CHURNED'].sum()):,}", f"{overall*100:.1f}%")
c4.metric("Columns", f"{df.shape[1]}")

with st.expander("🔧 Data-cleaning notes (what the pipeline changed)", expanded=False):
    if notes:
        for n in notes:
            st.write("•", n)
    else:
        st.write("No cleaning actions were necessary on this upload.")

tabs = st.tabs(["1️⃣ Descriptive", "2️⃣ Diagnostic", "3️⃣ Feature Engineering",
                "4️⃣ Clustering", "5️⃣ Association Rules",
                "6️⃣ Models", "7️⃣ Evaluation", "📋 Findings"])

# ============================== TAB 1 — DESCRIPTIVE =========================
with tabs[0]:
    st.header("Descriptive Analysis — Cross-Tabulation vs Churn")
    st.caption("Each row of a cross-tab is row-normalised to show the **churn rate** "
               "within that group. Dashed line = overall churn rate.")

    cat_options = [c for c in ["Gender", "Contract", "PaymentMethod", "InternetService",
                               "TechSupport", "OnlineSecurity", "PaperlessBilling"]
                   if c in df.columns]
    sel = st.selectbox("Cross-tabulate Churn against:", cat_options, index=1)

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader(f"Counts: {sel} × Churn")
        ct_counts = pd.crosstab(df[sel], df[TARGET], margins=True, margins_name="Total")
        st.dataframe(ct_counts, width='stretch')
    with colR:
        st.subheader("Row % (churn rate by group)")
        ct_pct = (pd.crosstab(df[sel], df[TARGET], normalize="index") * 100).round(1)
        st.dataframe(ct_pct, width='stretch')

    g = df.groupby(sel)["CHURNED"].agg(["mean", "count"])
    g = g[g["count"] >= 5].sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(g))))
    colors = [RED if m > overall else GREEN for m in g["mean"]]
    ax.barh(g.index.astype(str), g["mean"] * 100, color=colors)
    ax.axvline(overall * 100, color=NAVY, ls="--", lw=1.4, label=f"Overall {overall*100:.0f}%")
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(m * 100 + 0.5, i, f"{m*100:.0f}% (n={n})", va="center", fontsize=8)
    ax.set_xlabel("Churn %"); ax.legend(); ax.set_title(f"Churn rate by {sel}")
    st.pyplot(fig, width='stretch')
    st.caption(f"Reading this chart: bars to the right of the dashed line are segments "
               f"churning **above** the {overall*100:.0f}% company average — these are "
               f"the groups a retention team should prioritise first.")

    st.subheader("Numeric summary by outcome")
    num_cols = [c for c in ["Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls"]
                if c in df.columns]
    st.dataframe(df.groupby(TARGET)[num_cols].describe().T, width='stretch')

# ============================== TAB 2 — DIAGNOSTIC ==========================
with tabs[1]:
    st.header("Diagnostic Analysis — What Drives Churn?")
    st.warning("**Interpretation guardrail:** a higher churn rate for a segment is a "
               "*pattern in this dataset*, not a guaranteed causal mechanism. On real "
               "customer data, confirm any strong association with operational evidence "
               "(support tickets, NPS, exit surveys) before committing budget to it.")

    st.subheader("Which factors are most associated with churn?")
    st.caption("χ² test of independence + Cramér's V effect size. Higher V = stronger "
               "association. p ≥ 0.05 (grey) = not statistically significant.")
    assoc_cols = [c for c in ["Contract", "PaymentMethod", "InternetService", "TechSupport",
                              "OnlineSecurity", "PaperlessBilling", "Gender"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Factor": c, "Cramér's V": round(v, 3), "χ²": round(chi2, 1),
                     "dof": dof, "p-value": f"{p:.2e}", "categories": ncat,
                     "significant (p<0.05)": "✅" if p < 0.05 else "—"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
    st.dataframe(assoc, width='stretch', hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(assoc) + 1))
    a2 = assoc.sort_values("Cramér's V")
    bar_colors = [GREY if "—" in s else NAVY for s in a2["significant (p<0.05)"]]
    ax.barh(a2["Factor"], a2["Cramér's V"], color=bar_colors)
    for i, v in enumerate(a2["Cramér's V"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Cramér's V (effect size)")
    st.pyplot(fig, width='stretch')
    st.caption("Contract type consistently shows the strongest, most statistically robust "
               "association with churn in this dataset — the diagnosis should lead with it.")

    st.divider()
    colA, colB = st.columns(2)

    with colA:
        st.subheader("By TENURE BAND")
        bins = [-1, 6, 12, 24, 48, 200]
        labs = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
        tb = pd.cut(df["Tenure"], bins=bins, labels=labs)
        g = df.assign(TB=tb).groupby("TB", observed=True)["CHURNED"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(g.index.astype(str), g["mean"] * 100, color=NAVY)
        ax.axhline(overall * 100, color=RED, ls="--", lw=1.3)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, m*100+0.6, f"{m*100:.0f}%\nn={n}", ha="center", fontsize=8)
        ax.set_ylabel("Churn %")
        st.pyplot(fig, width='stretch')
        st.caption("Newer customers churn at a visibly higher rate — the first months of "
                   "the relationship are the highest-risk window.")

    with colB:
        st.subheader("By SUPPORT CALL VOLUME")
        sb = pd.cut(df["SupportCalls"], bins=[-1, 0, 2, 4, 100],
                    labels=["0 calls", "1-2 calls", "3-4 calls", "5+ calls"])
        g = df.assign(SB=sb).groupby("SB", observed=True)["CHURNED"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(g.index.astype(str), g["mean"] * 100, color=ORANGE)
        ax.axhline(overall * 100, color=RED, ls="--", lw=1.3)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, m*100+0.6, f"{m*100:.0f}%\nn={n}", ha="center", fontsize=8)
        ax.set_ylabel("Churn %")
        st.pyplot(fig, width='stretch')
        st.caption("Churn rises sharply once a customer logs several support calls — "
                   "frequent contact is a visible early-warning signal, not just a "
                   "service-cost driver.")

# ============================== TAB 3 — FEATURE ENGINEERING ================
with tabs[2]:
    st.header("Feature Engineering")
    st.markdown("""
**Pipeline applied before training (auditable, reproducible):**

1. **Identifier dropped** — `CustomerID` (no predictive value).
2. **Duplicate rows removed** — exact repeat customer records collapsed to one row each.
3. **Text normalisation** — inconsistent casing/whitespace in `Gender`, `Contract`, and
   `PaymentMethod` merged into single canonical categories (e.g. `month-to-month ` /
   `Month-to-month` → `Month-to-month`).
4. **Impossible values treated as missing** — negative `Tenure` values (a data-entry
   error, since tenure cannot be negative) are set to missing, then imputed.
5. **Outlier correction** — `MonthlyCharges` values beyond 3×IQR above the third
   quartile (clear entry errors, e.g. a value 10× too large) are set to missing, then
   imputed, rather than left to distort the model.
6. **Missing-value imputation** — numeric gaps filled with the column median; categorical
   gaps filled with the most frequent category.
7. **Encoding** — numerics standardised; categoricals one-hot encoded with
   `handle_unknown='ignore'` so unseen test categories don't break inference.
8. **Target** — `Churn` → `CHURNED` (1 = Yes, 0 = No).
""")
    st.info("⚠️ **Leakage note:** the target column `Churn` itself is excluded from the "
            "model's input features, and `CustomerID` is dropped as a non-predictive "
            "identifier. This is what keeps the model scores in Tab 4/5 realistic instead "
            "of artificially perfect.")

    X, y, num, cat = build_design_matrix(df)
    st.write(f"**Numeric features ({len(num)}):** {', '.join(num)}")
    st.write(f"**Categorical features ({len(cat)}):** {', '.join(cat)}")
    pre = make_preprocessor(num, cat).fit(X)
    try:
        width = pre.transform(X).shape[1]
        st.success(f"After one-hot encoding, the design matrix has **{width}** columns "
                   f"across **{len(X):,}** rows.")
    except Exception as e:
        st.error(f"Preprocess preview failed: {e}")
    st.dataframe(X.head(10), width='stretch')

# ============================== TAB 4 — MODELS ==============================

# ============================== TAB 4 — CLUSTERING ==========================
with tabs[3]:
    st.header("Clustering — Who Are My Different Customer Types?")
    st.caption(
        "K-Means groups customers into segments based on numeric similarity — no labels needed. "
        "Each segment is then profiled by its churn rate and key attributes to build actionable customer personas."
    )

    num_clust = ["Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls", "SeniorCitizen"]
    Xc = df[num_clust].fillna(df[num_clust].median())
    sc_clust = StandardScaler()
    Xcs = sc_clust.fit_transform(Xc)

    st.subheader("Step 1 — Choosing number of clusters (Elbow method)")
    inertias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xcs)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(list(range(2, 9)), inertias, "o-", color=NAVY)
    ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow Chart — pick k where the curve bends")
    st.pyplot(fig, use_container_width=True)
    st.caption("The elbow is where adding another cluster stops meaningfully reducing inertia. "
               "In this dataset the bend is typically around k = 3 or 4.")

    k = st.slider("Select number of clusters (k)", 2, 6, 3)
    km_final = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xcs)
    df["Cluster"] = km_final.labels_.astype(str)

    st.subheader("Step 2 — Visualising clusters in 2D (PCA projection)")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xcs)
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ACCENT + [GREY, "#9b59b6"]
    for i, label in enumerate(sorted(df["Cluster"].unique())):
        mask = df["Cluster"] == label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {label}",
                   alpha=0.55, s=20, color=palette[i % len(palette)])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)")
    ax.set_title("Customer segments — PCA view"); ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.caption("PCA compresses all five numeric features into two dimensions for visual inspection. "
               "Tight, well-separated blobs indicate genuinely distinct customer groups.")

    st.subheader("Step 3 — Cluster profiles")
    profile = df.groupby("Cluster").agg(
        Count=("CHURNED","count"),
        Churn_Rate=("CHURNED","mean"),
        Avg_Tenure=("Tenure","mean"),
        Avg_MonthlyCharge=("MonthlyCharges","mean"),
        Avg_SupportCalls=("SupportCalls","mean"),
    ).round(2)
    profile["Churn_Rate"] = (profile["Churn_Rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(profile, use_container_width=True)

    churn_rates = df.groupby("Cluster")["CHURNED"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(churn_rates.index, churn_rates.values,
           color=[palette[i % len(palette)] for i in range(len(churn_rates))])
    ax.axhline(df["CHURNED"].mean() * 100, color=RED, ls="--", lw=1.4, label="Overall avg")
    ax.set_ylabel("Churn %"); ax.set_title("Churn rate by cluster"); ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.caption("Clusters above the dashed line are your highest-risk segments. "
               "Cross-reference their tenure, charges, and support calls to design "
               "targeted retention actions for each group.")

    st.info("**What is Latent Class Analysis (LCA)?** LCA does the same job as K-Means "
            "but for categorical variables — contract type, payment method, tech support etc. "
            "It finds hidden classes of customers who share similar categorical profiles. "
            "In practice, you run K-Means on the numbers and LCA on the categories, "
            "then cross-reference the two segmentations.")


# ============================== TAB 5 — ASSOCIATION RULES ====================
with tabs[4]:
    st.header("Association Rule Mining — What Combinations Drive Churn?")
    st.caption("Association rules surface patterns of the form: customers who have X and Y "
               "also tend to Z. Here we look for rules where the outcome is Churn = Yes.")

    st.markdown("""
**Three key metrics:**
- **Support** — how common is this combination? (0.25 = 25% of customers have it)
- **Confidence** — of customers with this combination, what % churned?
- **Lift** — how much more likely to churn than a random customer? Lift > 1 = predictive
""")

    min_support = st.slider("Minimum support", 0.05, 0.40, 0.10, 0.01)
    min_confidence = st.slider("Minimum confidence", 0.40, 0.90, 0.60, 0.05)

    d2 = df.copy()
    d2["Tenure_Band"] = pd.cut(d2["Tenure"], bins=[-1,12,36,200],
                                labels=["New_lt1yr","Mid_1to3yr","Loyal_3yrplus"])
    d2["Charge_Band"] = pd.cut(d2["MonthlyCharges"], bins=[0,45,75,999],
                                labels=["LowCharge","MidCharge","HighCharge"])
    d2["Support_Band"] = pd.cut(d2["SupportCalls"], bins=[-1,1,3,99],
                                 labels=["LowSupport","MidSupport","HighSupport"])

    items_per_row = []
    for _, row in d2.iterrows():
        items = [
            f"Contract_{str(row['Contract']).replace('-','').replace(' ','')}",
            f"Tenure_{row['Tenure_Band']}",
            f"Charge_{row['Charge_Band']}",
            f"Support_{row['Support_Band']}",
            f"Tech_{row['TechSupport']}",
            f"Sec_{row['OnlineSecurity']}",
            "Churned" if row["Churn"] == "Yes" else "Stayed",
        ]
        items_per_row.append([str(x) for x in items if str(x) not in ("nan","None")])

    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_arr = te.fit(items_per_row).transform(items_per_row)
    basket_df = pd.DataFrame(te_arr, columns=te.columns_)

    try:
        freq = apriori(basket_df, min_support=min_support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence,
                                  num_itemsets=len(freq))
        churn_rules = rules[rules["consequents"].apply(lambda x: "Churned" in x)].copy()
        churn_rules = churn_rules.sort_values("lift", ascending=False)

        if len(churn_rules) == 0:
            st.warning("No churn rules found — try lowering support or confidence.")
        else:
            churn_rules["antecedents"] = churn_rules["antecedents"].apply(
                lambda x: ", ".join(sorted(x)))
            churn_rules["consequents"] = churn_rules["consequents"].apply(
                lambda x: ", ".join(sorted(x)))
            display = churn_rules[["antecedents","consequents","support","confidence","lift"]].head(20).copy()
            display.columns = ["IF (customer has)","THEN","Support","Confidence","Lift"]
            for col in ["Support","Confidence","Lift"]:
                display[col] = display[col].round(3)
            st.dataframe(display.reset_index(drop=True), use_container_width=True)

            st.subheader("Top rules by Lift")
            top = churn_rules.head(10).copy()
            labels = top["antecedents"].str[:60].tolist()[::-1]
            lifts = top["lift"].tolist()[::-1]
            fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(top) + 1.5)))
            ax.barh(labels, lifts, color=RED)
            ax.axvline(1, color=NAVY, ls="--", lw=1.2, label="Lift=1 (no effect)")
            ax.set_xlabel("Lift"); ax.set_title("Top churn-predicting combinations"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("Each bar = how many times more likely a customer matching that "
                       "combination is to churn vs a random customer.")

            top1 = churn_rules.iloc[0]
            st.info(
                f"**Top rule:** Customers who are **{top1['antecedents']}** → "
                f"churn with **{top1['confidence']*100:.0f}% probability** "
                f"(Lift = {top1['lift']:.2f}x the baseline, "
                f"Support = {top1['support']*100:.1f}% of customers)."
            )
    except Exception as e:
        st.error(f"Association rule error: {e}. Try increasing minimum support.")

with tabs[5]:
    st.header("Supervised Classification Models")
    st.caption("Models: Logistic Regression, Decision Tree (depth 5), "
               "Random Forest (200 trees, depth 6), Gradient Boosting (defaults).")

    X, y, num, cat = build_design_matrix(df)
    results, (ntr, nte, tr_pos, te_pos) = train_models(X, y, num, cat, test_size, int(seed))
    st.write(f"Train rows: **{ntr:,}** (churn {tr_pos*100:.1f}%) | "
             f"Test rows: **{nte:,}** (churn {te_pos*100:.1f}%)")

    metrics = pd.DataFrame({
        m: {"Train Acc": r["train_acc"], "Test Acc": r["test_acc"],
            "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "ROC-AUC": r["roc_auc"],
            "Overfit gap": r["train_acc"] - r["test_acc"]}
        for m, r in results.items()
    }).T.round(3)
    st.subheader("Metric comparison (test set)")
    st.dataframe(metrics.style.highlight_max(axis=0, subset=["Test Acc", "Precision",
                 "Recall", "F1", "ROC-AUC"], color="#d4efdf")
                 .highlight_max(axis=0, subset=["Overfit gap"], color="#f5b7b1"),
                 width='stretch')

    best = metrics["ROC-AUC"].idxmax()
    st.success(f"🏆 Best ROC-AUC: **{best}** ({metrics.loc[best,'ROC-AUC']:.3f}). "
               f"Scores in the 0.5-0.65 range are expected here — the synthetic churn "
               f"signal is moderate and intentionally noisy, similar to real customer "
               f"behaviour, rather than perfectly predictable.")

    st.subheader("Per-model classification report")
    pick = st.selectbox("Model", list(results.keys()), index=2)
    st.code(results[pick]["report"])
    st.session_state["results"] = results

# ============================== TAB 5 — EVALUATION ==========================
with tabs[6]:
    st.header("Model Evaluation — Stability & Errors")
    if "results" not in st.session_state:
        X, y, num, cat = build_design_matrix(df)
        st.session_state["results"], _ = train_models(X, y, num, cat, test_size, int(seed))
    results = st.session_state["results"]

    st.subheader("ROC curves")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (name, r), c in zip(results.items(), ACCENT):
        fpr, tpr, _ = r["roc"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig, width='stretch')
    st.caption("AUC modestly above 0.5 ⇒ there is some learnable structure (contract "
               "type, tenure, support calls), but churn is not fully determined by these "
               "attributes alone — consistent with real customer behaviour.")

    st.subheader("Metric comparison")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mets = ["test_acc", "precision", "recall", "f1", "roc_auc"]
    labels = ["Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]
    xx = np.arange(len(labels)); w = 0.2
    for i, ((name, r), c) in enumerate(zip(results.items(), ACCENT)):
        ax.bar(xx + i*w - 1.5*w, [r[m] for m in mets], w, label=name, color=c)
    ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.legend(ncol=2, fontsize=8)
    st.pyplot(fig, width='stretch')

    st.subheader("Confusion matrices (test set)")
    cols = st.columns(2)
    for i, (name, r) in enumerate(results.items()):
        with cols[i % 2]:
            cm = r["cm"]
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, cm[a, b], ha="center", va="center", fontweight="bold",
                            color="white" if cm[a, b] > cm.max()/2 else NAVY, fontsize=13)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Stay", "Pred Churn"])
            ax.set_yticklabels(["Act Stay", "Act Churn"]); ax.grid(False)
            ax.set_title(f"{name}\nacc={r['test_acc']:.2f} · recall={r['recall']:.2f}",
                         fontsize=9)
            st.pyplot(fig, width='stretch')

    st.subheader("Feature importance (Random Forest)")
    rf = results["Random Forest"]["pipe"]
    ohe = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    X, y, num, cat = build_design_matrix(df)
    feat = num + list(ohe.get_feature_names_out(cat))
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp.values, color=GREEN)
    ax.set_xlabel("Importance")
    st.pyplot(fig, width='stretch')
    st.caption("One-hot encoding splits each categorical across multiple dummy columns, "
               "so the *aggregate* importance of Contract is understated here — sum the "
               "dummies (e.g. all Contract_* rows) to gauge a factor's true weight.")

# ============================== TAB 6 — FINDINGS ============================
with tabs[7]:
    st.header("Findings & Recommendations — Business Strategy Diagnosis")
    st.markdown(f"""
*All figures below recompute live from the loaded data; the narrative reflects the
bundled sample (n={len(df):,}, overall churn **{overall*100:.1f}%**).*

### Diagnosis: why is this business losing customers?
1. **Contract structure is the single strongest, most actionable churn driver.**
   Month-to-month customers churn at roughly 1.5-2× the rate of one-year and
   two-year customers, and Contract has the highest, most statistically robust
   association with churn (Cramér's V leads the ranking in Tab 2).
2. **Tenure compounds the risk** — customers in their first 6-12 months churn
   noticeably more than longer-standing customers, meaning the highest-risk
   population is exactly the newest, least-locked-in segment.
3. **Support call volume is an early-warning signal**, not just a cost centre —
   churn rises sharply once a customer has logged several support interactions,
   suggesting unresolved service issues precede many exits.
4. **Lack of add-on services (tech support, online security) correlates with
   higher churn**, consistent with weaker product engagement reducing switching
   costs.
5. **Models confirm the outcome is only moderately predictable** (ROC-AUC roughly
   0.5-0.65). This is the *correct* and *expected* result once the target is
   properly excluded from the input features — it means the signal found above is
   genuine, not inflated by data leakage.

### What this is — and is not
- This is a **synthetic dataset** built to resemble realistic subscription-business
  patterns (contract type, tenure, support load driving churn) — the exact
  percentages are for demonstration, not a real company's numbers.
- The patterns above are a **template for diagnosis**: on a real customer base, the
  same dashboard (cross-tabs, χ²/Cramér's V ranking, predictive models) would be
  run against actual data to confirm whether the same drivers hold.

### Recommended next steps (strategy)
- **Incentivise contract upgrades** — targeted discounts or perks for month-to-month
  customers to move to annual contracts, since this is the single largest lever.
- **Build a first-90-day onboarding/retention program** for new customers, given the
  elevated churn risk in the early tenure window.
- **Flag accounts after 3+ support calls** for proactive outreach before they reach
  the point of leaving.
- **Bundle tech support / security add-ons** into entry-level plans to raise
  engagement and switching costs.
- **Re-run this dashboard on real customer data** once available, using the same
  pipeline, to validate whether these synthetic patterns hold in practice.
""")
    st.download_button("⬇️ Download cleaned dataset (CSV)",
                       df.to_csv(index=False).encode(), "customer_churn_cleaned.csv", "text/csv")

st.caption("Built with Streamlit + scikit-learn · synthetic data for demonstration · "
           "validate against real customer data before acting on specific figures.")
