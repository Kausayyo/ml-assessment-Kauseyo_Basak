# ML Assessment — Submission Guide

> **How to use this README:** Read the repo structure first, then follow the step-by-step walkthrough for each question. Every step explains *what* was done, *why* it was done, and *what the output means*. Results shown below are from the already-executed notebooks — you don't need to re-run anything.

---

## Repository Structure

```
ml-assessment-<your-name>/
├── README.md                        ← You are here
├── part_a/
│   ├── q1_supervised.ipynb          ← Heart disease classification (28 marks)
│   ├── q2_unsupervised.ipynb        ← Customer segmentation (22 marks)
│   └── q3_feature_engineering.ipynb ← Retail regression pipeline (20 marks)
├── part_b/
│   └── business_analysis.md         ← Business case written answers (30 marks)
└── data/
    ├── q1_heart_disease.csv
    ├── q2_customers.csv
    └── q3_retail_promotions.csv
```

> ⚠️ All notebooks are **already executed** — every cell has its output saved. Graders do not need to run them.

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

---

# Q1 — Supervised Learning: Heart Disease Classification

**File:** `part_a/q1_supervised.ipynb`  
**Dataset:** `data/q1_heart_disease.csv` — 800 rows × 12 columns  
**Goal:** Predict `heart_disease` (1 = present, 0 = absent)

---

### Step 1 — Data Loading and Inspection

**What we do:** Load the CSV with pandas, print shape, data types, missing value counts, and the first 5 rows.

**Why:** Before any modelling, you need to understand what you're working with — how many rows/columns, whether features are numerical or categorical, and where data is missing.

**Key findings:**
```
Shape: (800, 12)
Missing values → resting_bp: 24, cholesterol: 32  (all others: 0)
Categorical columns: chest_pain_type, resting_ecg, st_slope
```

---

### Step 2 — Exploratory Data Analysis (3 charts + heatmap)

**What we do:** Produce four visualisations saved as `q1_eda_plots.png` and `q1_corr_heatmap.png`.

| Chart | What it shows | Key insight |
|---|---|---|
| **Target class distribution** (bar) | Count of disease vs no-disease | ~409 disease vs ~391 no-disease — nearly balanced, no resampling needed |
| **Age histogram by class** (overlapping hist) | Age spread for each class | Disease-present patients skew slightly older |
| **Max HR boxplot by class** | Exercise heart rate by class | Lower max HR strongly associated with disease — key predictor |
| **Correlation heatmap** | Linear relationships between all numerical features | `oldpeak` (~0.5) and `max_hr` (~-0.4) are the strongest correlates with target |

**Why:** Visualisations inform preprocessing choices (e.g., do we need to handle outliers?) and reveal which features are likely to matter. Each chart is followed by a markdown interpretation cell in the notebook.

---

### Step 3 — Data Preprocessing

**What we do:** Handle missing values → encode categoricals → scale numericals → split data.

#### 3a. Missing Value Imputation
```python
df = df.assign(
    resting_bp=df['resting_bp'].fillna(df['resting_bp'].median()),
    cholesterol=df['cholesterol'].fillna(df['cholesterol'].median())
)
```
**Strategy: Median imputation**  
**Why median, not mean?** Medical data often contains physiological outliers (e.g., hypertensive crises, extreme cholesterol readings). The mean is sensitive to these extremes — median is not.  
**Why not drop rows?** We only have 800 rows. Dropping 56 rows (~7%) would meaningfully shrink the training set.

#### 3b. One-Hot Encoding
```python
cat_cols = ['chest_pain_type', 'resting_ecg', 'st_slope']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
```
Converts string categories into binary indicator columns. `drop_first=False` retains all dummies for interpretability.

#### 3c. Standard Scaling
```python
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])
```
Scales `age`, `resting_bp`, `cholesterol`, `max_hr`, `oldpeak` to zero mean and unit variance. Required for models sensitive to feature magnitude.

#### 3d. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
- `stratify=y` preserves the 51/49 class ratio in both train and test sets
- `random_state=42` ensures reproducibility
- Result: **640 training rows, 160 test rows**

---

### Step 4 — Model Training

**What we do:** Train three classifiers, all with `random_state=42`.

| Model | Key characteristic |
|---|---|
| `DecisionTreeClassifier` | Single tree — simple, interpretable, prone to overfitting |
| `RandomForestClassifier` | 100 trees averaged — reduces variance via bagging |
| `GradientBoostingClassifier` | Sequential error correction — typically highest accuracy |

All fit on `X_train`, evaluated on `X_test`.

---

### Step 5 — Model Evaluation

**What we do:** Print confusion matrix and full classification report for each model. Confusion matrices are saved as `q1_confusion_matrices.png`.

**Actual results from executed notebook:**

| Model | Precision (Disease) | Recall (Disease) | F1 (Disease) | Accuracy |
|---|---|---|---|---|
| Decision Tree | 0.72 | 0.73 | 0.72 | 72% |
| Random Forest | 0.78 | 0.81 | **0.80** | 79% |
| Gradient Boosting | 0.78 | 0.78 | 0.78 | 78% |

**Best model: Random Forest** — highest F1-score (0.80) and Recall (0.81) for the positive class.

**Why F1 over accuracy?** In medical classification, a false negative (missing a disease) is clinically dangerous. F1-score captures the trade-off between Precision and Recall, making it more meaningful than accuracy alone. Random Forest's high Recall (0.81) means it correctly identifies 81% of actual disease cases.

---

### Step 6 — Hyperparameter Tuning

**What we do:** Run `GridSearchCV` on `GradientBoostingClassifier` (our runner-up with consistent performance and more tunable hyperparameters than RF defaults).

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None],
    'learning_rate': [0.05, 0.1]
}
grid_search = GridSearchCV(..., cv=5, scoring='f1')
```

**Actual results:**
```
Best Parameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
Best CV F1 Score: 0.8329

Tuned GB → F1 (Disease): 0.79  |  Accuracy: 79%
Baseline GB → F1 (Disease): 0.78  |  Accuracy: 78%
```

**Interpretation:** A lower learning rate (`0.05`) with shallow trees (`max_depth=3`) prevents overfitting — the model takes smaller, more conservative gradient steps. The tuned model achieves a marginal improvement over the baseline, confirming the default settings were already close to optimal for this dataset size.

---

---

# Q2 — Unsupervised Learning: Customer Segmentation

**File:** `part_a/q2_unsupervised.ipynb`  
**Dataset:** `data/q2_customers.csv` — 500 rows × 6 columns  
**Goal:** Segment customers using K-Means, visualise with PCA

---

### Step 1 — Data Preparation

**What we do:** Load data, confirm no missing values, apply StandardScaler.

```
Shape: (500, 6) | Missing values: 0
Features: age, annual_spend, visits_per_month, basket_size, 
          days_since_last_visit, num_categories_purchased
```

**Why scaling is essential for K-Means:**  
K-Means assigns clusters by computing **Euclidean distance** between points. Without scaling, features with large numeric ranges (e.g., `annual_spend` in thousands) will completely dominate the distance calculation, drowning out low-range features like `visits_per_month`. StandardScaler transforms every feature to mean=0, std=1 so all features contribute equally.

---

### Step 2 — Choosing K (Elbow Method)

**What we do:** Compute WCSS (Within-Cluster Sum of Squares) for K = 1 through 10, plot the elbow curve, saved as `q2_elbow.png`.

**Why WCSS?** WCSS measures the total variance within clusters — lower is better, but it always decreases as K increases. The "elbow" is where adding more clusters yields diminishing returns.

**Result:** The elbow occurs at **K = 4** — the curve's descent steepens up to K=4, then flattens. Going beyond K=4 produces marginal within-cluster improvement without meaningful new business segments.

---

### Step 3 — K-Means Clustering

**What we do:** Fit K-Means with K=4, add cluster labels, print centroids in original scale.

**Actual cluster sizes:**
```
Cluster 0: 170 customers
Cluster 1:  80 customers
Cluster 2: 165 customers
Cluster 3:  85 customers
```

**Cluster centroids (original scale):**

| Cluster | Age | Annual Spend | Visits/Month | Basket Size | Days Since Visit | Business Label |
|---|---|---|---|---|---|---|
| **0** | 25 | £14,847 | 14 | £559 | Low | 🟦 Young Frequent Budget Shoppers |
| **1** | 57 | £89,814 | 2 | £5,296 | High | 🟥 Older High-Value Occasional Buyers |
| **2** | 40 | £43,341 | 8 | £2,022 | Medium | 🟩 Mid-Tier Regular Customers |
| **3** | 57 | £89,036 | 3 | £5,751 | High | 🟪 Older High-Value Low-Frequency |

**Business interpretation:**
- **Cluster 0** — High visit frequency, low spend per visit. Young shoppers browsing often but buying small. Target with upsell promotions and bundle offers.
- **Cluster 1 & 3** — High annual spend, infrequent visits. VIP customers who buy in bulk when they do shop. Target with exclusive loyalty rewards and early access offers.
- **Cluster 2** — Core reliable customer base. Moderate everything. Target with seasonal promotions to increase basket size.

---

### Step 4 — Dimensionality Reduction with PCA

**What we do:** Reduce 6 features to 2 principal components for visualisation.

**Actual results:**
```
PC1 explained variance: 83.56%
PC2 explained variance:  5.57%
Total explained:         89.13%
```

**Feature loadings:**

| Feature | PC1 | PC2 |
|---|---|---|
| age | 0.41 | -0.26 |
| annual_spend | **0.42** | -0.03 |
| visits_per_month | -0.41 | 0.21 |
| basket_size | 0.41 | -0.20 |
| days_since_last_visit | 0.38 | **0.91** |
| num_categories_purchased | 0.41 | -0.14 |

**Interpretation:**
- **PC1** (83.6% of variance) is the **overall customer value axis** — high positive loadings on `annual_spend`, `basket_size`, `age`, and `num_categories_purchased`. It separates high-value customers (right) from low-value ones (left).
- **PC2** (5.6% of variance) is dominated by `days_since_last_visit` — it separates **lapsed customers** (top) from **recently active** ones (bottom).

89% variance explained by just 2 components confirms the dataset has a strong underlying structure — the clusters are real, not noise.

---

### Step 5 — Cluster Visualisation

**What we do:** Scatter plot of PC1 vs PC2, coloured by cluster label, saved as `q2_cluster_viz.png`.

The plot shows four visually distinct groupings in 2D space, confirming the K=4 choice produces genuine, separable segments rather than arbitrary divisions.

---

---

# Q3 — Feature Engineering & Regression Pipeline

**File:** `part_a/q3_feature_engineering.ipynb`  
**Dataset:** `data/q3_retail_promotions.csv` — 1200 rows × 9 columns  
**Goal:** Predict `items_sold` using a scikit-learn Pipeline

---

### Step 1 — Date Feature Engineering

**What we do:** Parse `transaction_date` and extract 4 new features.

```python
df['year']        = df['transaction_date'].dt.year
df['month']       = df['transaction_date'].dt.month        # 1–12
df['day_of_week'] = df['transaction_date'].dt.dayofweek   # 0=Mon, 6=Sun
df['is_month_end'] = (df['transaction_date'].dt.day >= 25).astype(int)
```

**Why these features?**
- `month` captures **seasonality** — sales patterns differ by time of year
- `day_of_week` captures **weekly rhythms** — weekends may drive more footfall
- `is_month_end` captures **payday spending spikes** — the last week of the month when salaries arrive often sees higher discretionary spending
- `year` captures **year-over-year growth trends**

**Confirmed output (first 8 rows):**
```
transaction_date  year  month  day_of_week  is_month_end
2022-01-01        2022      1            5             0
2022-01-02        2022      1            6             0
...
```
Shape expands from (1200, 9) → **(1200, 13)** with 4 new columns.

---

### Step 2 — Temporal Train-Test Split

**What we do:** Sort by date, use first 80% as train and most recent 20% as test.

```python
df = df.sort_values('transaction_date').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]   # rows 0–959
test_df  = df.iloc[split_idx:]   # rows 960–1199
```

**Actual split:**
```
Training records: 960  (up to 2024-06-11)
Test records:     240  (from 2024-06-12 onward)
```

**Why NOT a random split for time-series data?**  
A random split allows future dates into training and past dates into testing. This is **data leakage** — the model learns seasonal/trend signals from the future, making test performance optimistically misleading. In production, a model only ever predicts the future from the past. A temporal split faithfully simulates this.

---

### Step 3 — Preprocessing Pipeline

**What we do:** Build a `ColumnTransformer` inside a `Pipeline` that handles categoricals and numericals differently.

```python
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
     ['promotion_type', 'location_type', 'store_size']),
    ('num', StandardScaler(), 
     ['competition_density', 'is_weekend', 'is_festival',
      'year', 'month', 'day_of_week', 'is_month_end'])
])
```

**Key design decisions:**
- **`handle_unknown='ignore'`** — if a promotion type appears in test but not train, it won't crash; it'll just produce all-zeros for that category
- **Pipeline approach** — the preprocessor is `.fit()` only on training data, then `.transform()` applied to both train and test. This prevents test data statistics from leaking into the scaler or encoder

---

### Step 4 — Model Training and Evaluation

**What we do:** Train two models inside full pipelines, evaluate with RMSE and MAE, plot parity charts and feature importances.

#### Models

```python
lr_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=200))])
```

#### Actual Results

| Model | RMSE | MAE |
|---|---|---|
| **Linear Regression** | **27.13** | **21.07** |
| Random Forest Regressor | 31.24 | 25.16 |

> **Surprising result:** Linear Regression outperforms Random Forest here. This suggests the relationship between features and `items_sold` is largely **linear** in this dataset — adding non-linear complexity via Random Forest actually introduces unnecessary variance on the limited test window (240 rows). Linear Regression's simplicity is an advantage when the signal is clean and linear.

#### Parity Plots
Saved as `q3_parity_plots.png`. A perfect model would have all points on the red diagonal line. Points scattered around it indicate model error. Linear Regression's points cluster more tightly around the diagonal.

#### Top 5 Feature Importances (Random Forest)

```
Feature                  Importance
is_festival              0.176   ← Festival days drive the biggest sales swings
store_size_small         0.164   ← Small stores behave very differently
location_type_urban      0.112   ← Urban location significantly affects volume
day_of_week              0.088   ← Day of week matters (weekends)
is_weekend               0.067   ← Weekend flag adds incremental signal
```

**Business insight:** `is_festival` is the single most important predictor — stores sell ~38% more items during festival periods. Store size and location type together account for ~28% of model importance, confirming that store context is crucial. This supports the recommendation in Part B for stratified or hierarchical modelling by store type.

---

---

# Part B — Business Case Analysis

**File:** `part_b/business_analysis.md`

| Section | Topic | Marks |
|---|---|---|
| B1(a) | ML problem formulation — target, features, problem type | 3 |
| B1(b) | Why items sold > revenue as target variable | 3 |
| B1(c) | Alternative to one global model across 50 stores | 2 |
| B2(a) | Joining 4 tables — grain, aggregations | 4 |
| B2(b) | EDA strategy — 4 analyses with interpretations | 4 |
| B2(c) | Handling 80% no-promotion imbalance | 2 |
| B3(a) | Train-test split design, metrics, and interpretation | 4 |
| B3(b) | Using feature importance + SHAP to explain different recommendations | 4 |
| B3(c) | End-to-end deployment: saving, serving, monitoring, retraining | 4 |

Open `part_b/business_analysis.md` for the complete written answers.

---

## Quick Reference — Files and Their Purpose

| File | Purpose | Marks |
|---|---|---|
| `part_a/q1_supervised.ipynb` | Heart disease classification — 3 models + tuning | 28 |
| `part_a/q2_unsupervised.ipynb` | K-Means segmentation + PCA visualisation | 22 |
| `part_a/q3_feature_engineering.ipynb` | Date engineering + regression pipeline | 20 |
| `part_b/business_analysis.md` | Written business case answers | 30 |
| `data/*.csv` | Source datasets loaded with relative paths | — |

**Total: 100 marks**
