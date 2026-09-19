# Transformer Fault Diagnosis via Dissolved Gas Analysis (DGA) + ML

Predicts transformer fault type from dissolved gas concentrations, and benchmarks a Random Forest against the two classical, industry-standard diagnostic methods — **Duval Triangle** and **IEC 60599 Basic Gas Ratio** — that real protection engineers use by hand.

## Status

- [x] Data loading and cleaning
- [x] Log-transform + feature engineering
- [x] EDA — gas distributions by fault type
- [x] PCA visualization
- [x] Random Forest model, 5-fold cross-validated
- [x] Confusion matrix + feature importance
- [x] Prediction function for new samples
- [x] Duval Triangle — full implementation, verified zone geometry
- [x] IEC 60599 Basic Gas Ratio — full implementation, verified thresholds
- [x] Classical vs. ML benchmark, including per-class breakdown

## The problem

Given concentrations (ppm) of 5 gases dissolved in transformer oil — H2, CH4, C2H6, C2H4, C2H2 — classify which of 6 fault types is occurring:

| Code | Meaning |
|---|---|
| PD | Partial Discharge |
| D1 | Low-Energy Discharge |
| D2 | High-Energy Discharge |
| T1 | Thermal fault, <300°C |
| T2 | Thermal fault, 300–700°C |
| T3 | Thermal fault, >700°C |

## Why this project isn't a generic classifier

Most "ML on a CSV" projects stop at reporting an accuracy number. This one implements the actual classical diagnostic rules — Duval Triangle and IEC 60599 Basic Gas Ratio, both derived from real IEEE/IEC reference material, not guessed — and benchmarks the ML model against them **on the same 589 rows, per fault class**. The result (see Section 10 below) is more interesting than "ML wins": the classical methods have real coverage gaps — they sometimes refuse to answer at all — on top of scoring lower where they do answer.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
The dataset ships inside `data/`, so no separate download is needed. Open `notebooks/01_dga_fault_diagnosis.ipynb`, select the venv kernel, and run top to bottom (the notebook already has all outputs saved, so you can also just read it on GitHub without running anything).

## Dataset

589 samples, 6 fault classes, no missing values. Reasonably balanced (60–149 samples/class):

```
D2: 149    T1: 111    T3: 104    D1: 91    PD: 74    T2: 60
```

The raw spreadsheet's label column header ships in Chinese (`故障类型`, "fault type") — the first line of code renames it to `fault_type` for readability. Everything downstream works on the English labels (`PD`, `D1`, `D2`, `T1`, `T2`, `T3`).

---

## How it works — code walkthrough

### 1. Load and clean
```python
df = pd.read_excel("../data/IEEE_STANDARD.xlsx")
df = df.rename(columns={'故障类型': 'fault_type'})
```
Loads the spreadsheet into a pandas DataFrame. Columns: `H2, CH4, C2H6, C2H4, C2H2, fault_type`.

### 2. Log-transform the gases
```python
gas_cols = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
for col in gas_cols:
    df[f'log_{col}'] = np.log1p(df[col])
```
Raw gas concentrations span ~5 orders of magnitude (0.0001 to 90,000+ ppm). Without compressing this, a handful of huge outlier readings would dominate any model or plot. `log1p(x) = log(1+x)` — used instead of plain `log` so near-zero readings (the 0.0001 "not detected" floor) stay near zero instead of blowing up to large negative numbers.

### 3. Feature engineering — ratios
```python
df['ratio_c2h2_c2h4'] = df['C2H2'] / df['C2H4'].replace(0, 0.0001)
df['ratio_ch4_h2']    = df['CH4']  / df['H2'].replace(0, 0.0001)
df['ratio_c2h4_c2h6'] = df['C2H4'] / df['C2H6'].replace(0, 0.0001)

feature_cols = [f'log_{c}' for c in gas_cols] + ['ratio_c2h2_c2h4', 'ratio_ch4_h2', 'ratio_c2h4_c2h6']
X = df[feature_cols]       # inputs: 589 rows x 8 columns
y = df['fault_type']       # answer key: 589 labels
```
These three ratios are the same ones the IEC 60599 and Duval methods are built on (Section 8/9 below) — giving the ML model access to the same domain signal the classical methods use, on top of the raw magnitudes.

### 4. PCA — visualize class separability
```python
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
```
Scaling first matters — PCA is sensitive to raw magnitude, and without it a feature with naturally bigger numbers would dominate regardless of how informative it actually is. PCA then finds 2 new axes that capture as much of the original 8-feature spread as possible.

**Result:** PC1 + PC2 together explain **52.5%** of total variance (35.7% + 16.9%). Just over half the structure is visible in the 2D plot — real, but interpret any overlap you see with that caveat in mind, since a meaningful chunk of the separating information lives in dimensions the plot can't show.

### 5. Train and evaluate — the honest way
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=skf)
```
`cross_val_predict` trains on 4 folds and predicts the 5th, five times over — every prediction is on data the model never trained on that round. This is what makes the resulting accuracy trustworthy, unlike `model.fit(X,y)` then scoring on the same `X` (which just measures memorization).

**Results:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| D1 | 0.70 | 0.63 | **0.66** | 91 |
| D2 | 0.80 | 0.89 | 0.84 | 149 |
| PD | 0.86 | 0.74 | 0.80 | 74 |
| T1 | 0.88 | 0.88 | 0.88 | 111 |
| T2 | 0.79 | 0.77 | 0.78 | 60 |
| T3 | 0.90 | 0.92 | **0.91** | 104 |

**Overall accuracy: 0.82 — macro F1: 0.81**

**Sanity baseline:** always guessing the majority class (D2, 149/589) gets 25.3% accuracy for free. 82% vs. 25.3% confirms the model is learning real signal, not noise.

**Interpretation:** T3 and T1 (thermal faults) are diagnosed most reliably — distinct gas signatures. D1 is the weakest class (F1 0.66) — this makes physical sense, since D1 (low-energy discharge) sits on the spectrum right between PD and D2, making it the natural confusion point.

### 6. Feature importance — the key finding
```python
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
```

**Results:**
```
ratio_c2h4_c2h6    0.187
ratio_c2h2_c2h4    0.185
ratio_ch4_h2       0.168
log_C2H2           0.133
log_C2H4           0.112
log_H2             0.085
log_C2H6           0.068
log_CH4            0.063
```

**This is the strongest single result in the project:** the top 3 features are all ratios, outranking every raw gas value. The model independently rediscovered — with zero rules programmed in — that *relationships between gases* matter more than *how much of any one gas* is present. That's the exact same principle Duval and IEC encode by hand. This is the sentence to lead with when explaining the project to someone.

### 7. `predict_fault()` — scoring a new sample
```python
def predict_fault(h2, ch4, c2h6, c2h4, c2h2, model, feature_cols):
    new_sample = pd.DataFrame([{...}])[feature_cols]   # [feature_cols] re-orders by NAME to match training order
    return model.predict(new_sample)[0], dict(zip(model.classes_, model.predict_proba(new_sample)[0]))
```
Builds the same 8 features used at training time from raw ppm values, then returns both the single best-guess label and the full per-class probability breakdown.

**Critical gotcha:** scikit-learn matches input columns by *position*, not name. The `[feature_cols]` at the end re-selects columns in the exact order used during training — skip this and you can get a silently wrong prediction with no error thrown.

`predict()` gives the single best guess. `predict_proba()` gives confidence across all 6 classes — reporting this instead of a bare label is more honest and more interview-worthy, since it shows the model's uncertainty, not false confidence.

### 8. Duval Triangle — `bary_to_xy()` and `duval_triangle()`

The Duval Triangle plots the relative percentages of three gases — CH4, C2H4, C2H2 — as one point on an equilateral triangle, split into 7 zones (6 fault types plus **DT**, a mixed thermal/electrical zone that isn't one of this dataset's 6 classes).

```python
def bary_to_xy(a, b, c):
    total = a + b + c
    a, b, c = a/total, b/total, c/total
    x = 0.5*(2*b + c)
    y = (np.sqrt(3)/2)*c
    return x, y
```
Converts the three percentages (which sum to ~100) into a 2D cartesian point, so a standard point-in-polygon test can be used instead of manual geometry per zone.

```python
def duval_triangle(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    ...
    for name, path in duval_paths.items():
        if path.contains_point((x, y)):
            return name
    # fallback: nearest zone centroid for boundary/rounding edge cases
```
Normalizes the three gases to percentages, converts to an (x, y) point, and checks which of the 7 zone polygons contains it (using `matplotlib.path.Path`). Falls back to the nearest zone centroid on the rare point that lands exactly on a boundary.

**Where the zone geometry came from:** the 7 zone polygons (vertex coordinates for PD, D1, D2, DT, T1, T2, T3) were pulled directly from the source code of a published, installable Python package (`duvals-triangle-plotter`) rather than hand-transcribed from a diagram — a single wrong vertex would silently invalidate every prediction near that boundary.

**Real results (tested on all 589 rows):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| D1 | 0.52 | 0.78 | 0.62 | 83 |
| D2 | 0.82 | 0.65 | 0.73 | 144 |
| PD | 0.85 | **0.42** | 0.56 | 67 |
| T1 | 0.81 | 0.51 | 0.62 | 108 |
| T2 | **0.36** | 0.51 | 0.42 | 59 |
| T3 | 0.71 | 0.96 | 0.82 | 102 |

Coverage: **563/589 (96%)** predicted one of the 6 real classes (the rest landed in the DT mixed-fault zone). Accuracy on those 563 rows: **65.7%**, macro F1 **0.629**. Over all 589 rows (DT counted as wrong, for a fair comparison to ML): **62.8%**.

**Weak spots:** PD recall is only 0.42 — the method frequently mistakes partial discharge for something else. T2 precision is only 0.36 — whenever Duval predicts T2, it's wrong almost two-thirds of the time.

### 9. IEC 60599 Basic Gas Ratio — `iec_ratio_method()`

This is the modern, IEC-adopted descendant of the 1978 Rogers Ratio table. It was chosen over plain Rogers because its 6 output categories line up exactly with this dataset's classes — Rogers' original table doesn't cleanly separate PD from D1/D2.

```python
def iec_ratio_method(h2, ch4, c2h6, c2h4, c2h2):
    r_c2h2_c2h4 = c2h2/c2h4 if c2h4 != 0 else float('inf')
    r_ch4_h2 = ch4/h2 if h2 != 0 else float('inf')
    r_c2h4_c2h6 = c2h4/c2h6 if c2h6 != 0 else float('inf')
    if r_ch4_h2 < 0.1 and r_c2h4_c2h6 < 0.2:
        return 'PD'
    ...
    return None   # ratio combination not covered by the standard table
```
Computes the three diagnostic ratios and checks them, in order, against the IEC 60599 standard's threshold table (sourced from an IEEE C57.104 / IEC 60599 reference document). Returns `None` when a ratio combination isn't covered by any row — a real limitation of the standard, not a bug, and exactly the coverage gap the benchmark below measures.

**Real results (tested on all 589 rows):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| D1 | 0.39 | 0.88 | 0.54 | 49 |
| D2 | **1.00** | **0.25** | 0.40 | 95 |
| PD | 0.74 | 0.74 | 0.74 | 23 |
| T1 | 0.84 | 0.90 | 0.87 | 70 |
| T2 | 0.85 | 0.96 | 0.90 | 53 |
| T3 | 0.91 | 0.89 | 0.90 | 87 |

Coverage: only **377/589 (64%)** of rows matched any row of the table at all — the rest fall into ratio combinations the standard's table simply doesn't anticipate. Accuracy on the 377 classified rows: **72.9%**, macro F1 **0.725**. Over all 589 rows (unmatched counted as wrong): **46.7%**.

**Interesting failure mode:** D2 precision is a perfect 1.00 but recall is only 0.25 — when the table commits to "D2," it's always right, but it's extremely conservative and misses 3 out of 4 real D2 cases (likely returning `None` for them instead).

### 10. The benchmark — `evaluate()` and the final comparison

```python
def evaluate(name, pred_col):
    covered = df[pred_col].isin(valid_classes)     # excludes IEC's None *and* Duval's DT zone
    filled = df[pred_col].where(covered, 'NO_MATCH')
    ...
```
Prints one summary line per method: how many of the 589 rows it actually classified as one of the 6 real fault types ("coverage"), accuracy restricted to those rows, and accuracy over the full dataset with everything else counted wrong — the last number is the fair, apples-to-apples comparison against ML, which always outputs one of the 6 classes.

**This is the project's actual result:**

| Method | Coverage | Accuracy (classified rows) | Accuracy (all 589 rows) | Macro F1 |
|---|---|---|---|---|
| IEC 60599 Ratio | 377/589 (64%) | 72.9% | 46.7% | 0.73 |
| Duval Triangle | 563/589 (96%) | 65.7% | 62.8% | 0.63 |
| **Random Forest (ML)** | 589/589 (100%) | — | **82.0%** | **0.81** |

The classical methods don't just score lower — they sometimes refuse to answer at all. IEC only classifies 64% of samples; the rest fall into ratio combinations the standard's table wasn't built to cover. Duval covers far more (96%) but is markedly weaker on specific classes (PD, T2). The Random Forest always outputs something, and it's the most accurate method regardless of how you slice the comparison. **The real thesis: classical rule-based methods have blind spots baked into fixed thresholds; a model trained on data doesn't.**

---

## Pushing this to GitHub

```bash
cd transformer-dga-diagnosis     # your project root (skip `git init` if you already have a repo here)
git init
git add .
git commit -m "Full DGA benchmark: RF (82% acc / 0.81 F1) vs IEC 60599 Ratio (46.7%) vs Duval Triangle (62.8%)"
```

Then on github.com: click **New repository**, name it (e.g. `transformer-dga-diagnosis`), leave it empty — **don't** tick "Add a README" (you already have one, and it'll create a merge conflict). Copy the repo URL it gives you, then:

```bash
git remote add origin <paste-your-repo-url-here>
git branch -M main
git push -u origin main
```

If it asks for credentials and password login fails (GitHub disabled that), you'll need a Personal Access Token instead of your password — GitHub will show a prompt/link for that the first time you push.

**Check `.gitignore` before your first commit** — it should already list `venv/`, `.ipynb_checkpoints/`, `__pycache__/` so you're not committing your entire virtual environment (that alone can be hundreds of MB and doesn't belong in the repo).
