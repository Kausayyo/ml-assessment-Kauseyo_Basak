# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target variable:** `items_sold` — the number of items sold at a store in a given month under a specific promotion.

**Candidate input features:**
- **Store attributes:** store size, location type (urban/semi-urban/rural), competition density, monthly footfall
- **Promotion attributes:** promotion type (Flat Discount, BOGO, Free Gift, Category-Specific Offer, Loyalty Points Bonus)
- **Temporal attributes:** month (captures seasonality), is_weekend, is_festival, is_month_end
- **Derived features:** interaction terms such as `promotion_type × location_type` and `promotion_type × store_size`
- **Historical features:** average items sold per store, promotion performance history per store

**Type of ML problem:** This is a **supervised regression problem**. The target variable (`items_sold`) is continuous. The goal is to learn a function *f(store, month, promotion) → items_sold* so that for any store-month combination, we can predict the outcome of each candidate promotion and recommend the one with the highest predicted value.

**Justification:** Since we want to select the best promotion per store per month, we could also frame this as a multi-class classification problem (predicting which promotion will win). However, regression is preferred because it outputs a magnitude (how many items sold), which allows the marketing team to compare promotions quantitatively (e.g., Flat Discount predicts 340 vs BOGO predicts 290), not just rank them. This richer output enables ROI-based decision-making.

---

### B1(b) — Why Items Sold Is a Better Target Than Revenue

**Revenue** is not a reliable optimisation target because it conflates price with volume. A promotion like a Flat Discount reduces per-item price while potentially increasing volume, so revenue could stay flat or decline even when the promotion is genuinely effective at moving product. Additionally, a BOGO offer yields different revenue per transaction versus a premium-priced Free Gift bundle, making cross-promotion comparisons misleading.

**Items sold (sales volume)** is a more direct measure of promotion effectiveness because:
1. It is price-agnostic — it captures whether a promotion actually drove customer behaviour (purchase decisions), not just price changes.
2. It aligns with retail objectives like inventory turnover, shelf space optimisation, and footfall conversion.
3. It is more stable to measure and less susceptible to confounding from pricing strategies applied outside of the ML model's scope.

**Broader principle:** In real-world ML projects, target variable selection should prioritise **measurability, controllability, and causal proximity to the action being modelled**. Revenue is influenced by many factors outside the model's control (pricing strategy, markdowns, returns). Items sold is more directly caused by promotion effectiveness. Choosing a target that is clean, causally linked to the intervention, and consistently measurable is fundamental to building models that produce actionable recommendations.

---

### B1(c) — Alternative to a Single Global Model

A single global model trained across all 50 stores assumes a uniform relationship between promotions and outcomes regardless of location. This is incorrect — a Flat Discount may outperform BOGO in rural stores (where price sensitivity is high) but underperform in urban flagship stores (where brand experience and variety matter more).

**Proposed strategy: Hierarchical / Grouped Modelling**

Train **separate models per location cluster** (urban, semi-urban, rural) or even **per store**, with the following design:

1. **Cluster-level base models:** Train one model per location type. This pools data within similar contexts, balancing between over-generalising (one global model) and over-fitting (one model per store with insufficient data).
2. **Store-level fine-tuning (if sufficient data):** For stores with 12+ months of data, fine-tune the cluster model using store-specific data via regularised regression or a gradient boosting model with store_id as a feature.
3. **Meta-learning / Mixed Effects:** Use a mixed-effects model or hierarchical Bayesian model that shares strength across stores (partial pooling), learning a global baseline and store-level deviations.

**Justification:** This accounts for the fact that the effect of a promotion (treatment) varies by store context (heterogeneous treatment effects). Ignoring this leads to averaged-out recommendations that are suboptimal for most individual stores.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

**Table grain and joins:**

| Table | Grain | Key |
|---|---|---|
| Transactions | 1 row = 1 transaction | `store_id`, `transaction_date` |
| Store Attributes | 1 row = 1 store | `store_id` |
| Promotion Details | 1 row = 1 promotion type | `promotion_type` |
| Calendar | 1 row = 1 date | `transaction_date` |

**Join sequence:**
1. Aggregate `transactions` to store-month level: `GROUP BY store_id, year-month` to compute `SUM(items_sold)` and identify the active `promotion_type` for that month.
2. `LEFT JOIN` store attributes on `store_id` to bring in size, location type, footfall, competition density.
3. `LEFT JOIN` promotion details on `promotion_type` to bring in promotion metadata (discount depth, cost, category scope).
4. `LEFT JOIN` calendar on `transaction_date` (or year-month) to bring in `is_weekend`, `is_festival`, and derived temporal flags.

**Final modelling dataset grain:** One row = one store × one month (with the promotion deployed that month and the resulting items sold). Each row contains all store attributes, the promotion used, temporal context, and the target `items_sold`.

**Aggregations before modelling:**
- Sum `items_sold` per store per month.
- Count transaction days to compute daily run rate (normalises for month length differences).
- Compute month-over-month sales change per store as a lag feature to capture trend.

---

### B2(b) — EDA Strategy

**1. Promotion effectiveness by location type (grouped bar chart)**
Plot mean items_sold for each promotion type, faceted or grouped by location type. *What to look for:* Does BOGO outperform Flat Discount in urban locations? Do rural stores respond more to price-based promotions? Findings directly inform whether a single model or stratified models are needed, and whether `promotion_type × location_type` interaction features should be engineered.

**2. Seasonality analysis (line chart: month vs mean items_sold)**
Plot average items_sold across months, overlaid by promotion type. *What to look for:* Are festival months (October–December) universally high regardless of promotion? Is there a January slump? If seasonal effects dominate, `month` and `is_festival` are critical features and the model must be trained with enough historical cycles to capture this.

**3. Store-level sales distribution (box plots by store size and location)**
Show the spread of items_sold across stores, grouped by store_size and location_type. *What to look for:* High variance within a group signals that store-level features (footfall, competition density) are important. Outlier stores should be investigated — they may have data quality issues or unique characteristics requiring special treatment.

**4. Promotion frequency and imbalance audit (pie chart + time-series heatmap)**
Check how often each promotion was used per store. *What to look for:* If a store only ever used one promotion type, we cannot estimate counterfactual effects from observational data alone. This influences whether causal inference techniques (e.g., propensity score weighting) are needed before modelling, and whether certain store-promotion combinations have too little data to trust predictions.

---

### B2(c) — Handling Promotion Imbalance (80% No-Promotion Transactions)

**Impact on the model:**
- **Biased learning:** If the model trains on predominantly no-promotion transactions, it will learn to predict the "no promotion" outcome as the default, under-estimating the lift from promotional activity.
- **Confounding:** Promotions may have been selectively deployed during high-footfall periods (e.g., festivals), making it appear that promotions cause high sales when in fact the high sales period caused the promotion deployment. This is **selection bias**.
- **Low signal for promotion effects:** The model may not see enough examples of each promotion type per store to learn reliable promotion-specific effects.

**Steps to address it:**
1. **Aggregate to store-month level first:** After aggregating transactions to one row per store-month (where each row has exactly one promotion type), the imbalance concern shifts to promotion-type frequency rather than transaction volume.
2. **Treat this as a causal inference problem:** Use **propensity score weighting** (inverse probability weighting) to up-weight records where less-common promotions were used, correcting for the fact that promotions were not randomly assigned.
3. **Separate modelling for promoted vs non-promoted periods:** Build a promotion uplift model that estimates the incremental effect of each promotion over the no-promotion baseline, rather than a model that conflates the two.
4. **Ensure minimum data thresholds:** Require a minimum number of historical observations per store-promotion combination before making a recommendation for that store; fall back to cluster-level estimates otherwise.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Strategy and Metrics

**Train-test split design:**
With 3 years of monthly data for 50 stores, we have approximately 1,800 store-month observations. The correct approach is a **temporal (walk-forward) split**:

- **Training set:** Month 1 through Month 30 (first 2.5 years)
- **Test set:** Month 31 through Month 36 (last 6 months)
- Optionally, use **rolling cross-validation** (train on months 1–12, test on month 13; train on months 1–13, test on month 14; etc.) to get stable estimates of model generalisation across different time windows.

**Why random split is inappropriate:** A random split breaks temporal ordering — future store-month rows can appear in the training set, allowing the model to implicitly learn from future trends, seasonal patterns, or promotion responses that would not be available at prediction time. This produces unrealistically optimistic evaluation metrics and models that underperform in deployment.

**Evaluation metrics and interpretation:**

| Metric | Formula | Business Interpretation |
|---|---|---|
| RMSE | √(mean(actual−predicted)²) | Average prediction error in units of items sold; penalises large errors heavily. A RMSE of 30 means typical predictions are ~30 items off. |
| MAE | mean(\|actual−predicted\|) | Average absolute error; easier to interpret as "on average, the model is off by X items per store per month." |
| MAPE | mean(\|actual−predicted\|/actual) × 100 | Percentage error; useful for communicating to non-technical stakeholders ("the model is within 8% of actual sales on average"). |
| R² | 1 − SS_res/SS_tot | Proportion of variance explained; an R² of 0.75 means the model accounts for 75% of the variation in items sold. |

For business reporting, MAE and MAPE are most intuitive. RMSE is used during model selection to penalise catastrophic errors.

---

### B3(b) — Explaining Different Recommendations for the Same Store

**Scenario:** Model recommends Loyalty Points Bonus for Store 12 in December but Flat Discount in March.

**Investigation approach using feature importance:**

1. **Identify top features globally:** Extract feature importances from the Random Forest. If `month`, `is_festival`, and `promotion_type` are the top 3 features, seasonal context is the primary driver.

2. **Local explanation with SHAP:** Use SHAP (SHapley Additive exPlanations) to generate **per-prediction feature contributions** for Store 12 in December and Store 12 in March. This reveals exactly which features pushed the prediction toward each promotion.
   - *December:* SHAP may show that `is_festival=1`, `month=12`, and `Loyalty_Points_Bonus=1` all contribute large positive values to the predicted items_sold, while `Flat_Discount=1` contributes negatively. This indicates the model learned that loyalty bonuses perform better during high-footfall festival months when customers are already inclined to buy.
   - *March:* SHAP may show `is_festival=0`, `month=3`, and that the model learned price sensitivity peaks in post-festival months — driving Flat Discount effectiveness.

3. **Communicating to the marketing team:**
   Present a side-by-side SHAP waterfall chart for each month. Frame it as: *"In December, the festival effect boosts customer engagement, making loyalty incentives most effective. In March, post-holiday budget consciousness means price-led promotions drive more units."* This translates model logic into business language the team already understands.

---

### B3(c) — End-to-End Deployment Process

**1. Saving the model:**
At the end of training, serialize the full scikit-learn Pipeline (preprocessor + model) using `joblib.dump(pipeline, 'promotion_model_v1.pkl')`. Store the model artifact in a versioned object store (e.g., AWS S3 with a version tag like `v1.2_2026-03`). Also persist the `scaler` and `encoder` state as part of the pipeline so that preprocessing is deterministic at inference time.

**2. Monthly data preparation and inference:**
At the start of each month:
- Pull the latest store attributes, forecasted footfall, upcoming festival flags, and store-competition data from the data warehouse.
- For each of the 50 stores, generate 5 candidate rows — one per promotion type — with that month's context features filled in.
- Load the saved model: `pipeline = joblib.load('promotion_model_v1.pkl')`.
- Run `pipeline.predict(candidate_rows)` → get predicted items_sold for each promotion per store.
- For each store, select the promotion with the highest predicted value and output the recommendation to a dashboard or directly to the marketing team's planning tool.

**3. Monitoring for model degradation:**
Implement the following monitoring layers:

| Check | Metric | Threshold | Action |
|---|---|---|---|
| **Prediction drift** | Mean and variance of predicted items_sold vs last 3 months | ±2 standard deviations | Alert; manual review |
| **Feature drift** | Distribution shift in input features (PSI score) | PSI > 0.2 for any key feature | Alert; investigate data pipeline |
| **Actuals vs predictions** | Monthly MAE and MAPE once actuals arrive (end of month) | MAPE degrades >5 pp vs baseline | Trigger retraining evaluation |
| **Business outcomes** | Actual items sold vs predicted for each recommended promotion | Sustained underperformance in >10 stores | Mandatory retraining |

**Retraining trigger:** Retrain the model quarterly using a rolling window of the most recent 18 months of data. If the monitoring system flags degradation mid-quarter (MAPE worsens by > 5 percentage points for 2 consecutive months), trigger an unscheduled retrain. Before promoting a new model version to production, run it in shadow mode for one month — generate its predictions without acting on them, and compare to the live model's accuracy using the actuals. Only promote if the new model is statistically better.
