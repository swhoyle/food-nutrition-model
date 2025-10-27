# Preprocessing Deliverable: Food Nutrition Model

**Team**: Spencer Hoyle, Francisco Chavezosa, Sean He  
**Dataset**: 424,297 products, 24 features  
**Goal**: Clean data for Nutri-Score prediction (A-E classification)


## The Problem

From our EDA we found some serious data quality issues:
- Energy values mixing kJ and kcal units (same canola oil showing 3,586 vs 900)
- Impossible outliers (10^30 range for carbs which is just wrong)
- 94% missing in added_sugars (basically useless column)
- 72% missing in labels_tags (but null just means "no labels")
- Class imbalance toward unhealthy (E=32%, D=23% vs A=15%, B=11%)

Good news is core nutrients are mostly there - sugar, fat, carbs, proteins missing less than 3%.


## Our Plan: 5 Phases

```
1. Clean data (duplicates, outliers, missing values, units)
2. Engineer features (ratios, binary flags from tags)
3. Encode & scale (make it model-ready)
4. Split data (train/val/test, stratified)
5. Balance classes (SMOTE on training set only)
```

**Important**: Phase 5 happens before Phase 4 because we split BEFORE balancing. If we balance first then split, synthetic samples leak into validation/test sets and we get fake performance metrics.


## Phase 1: Clean the Data

### Remove Duplicates

We found products like "Canola Harvest" appearing multiple times with different completeness scores. We'll group by brand+product and keep the one with highest completeness.

Why? Higher completeness means more reliable nutritional data.

### Cap Outliers

The extreme values are data entry errors not real food. We'll cap everything at 99th percentile to keep the reasonable data and clip the garbage.

**Capping thresholds**:
- Energy: ~1,000 kcal per 100g
- Fat: ~60g per 100g
- Carbohydrates: ~100g per 100g
- Sugars: ~80g per 100g
- Salt: ~5g per 100g
- Proteins: Remove any negative values

We'll calculate actual percentiles when implementing but those are reasonable upper bounds.

### Fix Mixed Units

Energy values are mixing kJ and kcal. We found canola oil at 3,586 which makes perfect sense in kJ (3,586 ÷ 4.184 ≈ 857 kcal).

**Fix**: Anything with energy over 500 is probably kJ. Convert by dividing by 4.184 so everything's in kcal per 100g.

### Handle Missing Values

Each column needs a different strategy based on WHY it's missing:

| Column | Missing % | Strategy | Reason |
|--------|-----------|----------|--------|
| added_sugars | 94% | Drop entirely | Too sparse, regulatory differences |
| labels_tags | 72% | Replace null with [] | Null = not reported, [] = no labels (semantic equivalence) |
| trans_fat | 41% | Fill with 0 | Regulatory baseline (if not reported, likely below threshold = 0) |
| brands | 28% | Fill with "unknown" | Different category, not mode |
| ingredients_from_palm_oil_n | 20% | Drop column | Already extracted as palm_oil_indicator from ingredients_analysis_tags in Phase 2 |
| Main nutrients | <3% | Fill with median | Robust to outliers |

**Why this strategy (and not more complex)?**

We're keeping it simple for the baseline:
- Could use KNN imputation for main nutrients (<3% missing) but median is faster and robust
- Could use mode for brands but that's misleading (pretending to know something we don't)
- Could use more sophisticated methods for trans_fat but regulatory 0 is a safe baseline
- ingredients_from_palm_oil_n is redundant since ingredients_analysis_tags already has palm oil info

Key insight is different missing reasons need different strategies. We're matching the strategy to the underlying cause of missingness, not using one-size-fits-all imputation.


## Phase 2: Feature Engineering

### Calculate Ratios

Raw values don't tell the full story - 30g sugar could be terrible (pure sugar) or fine (in a large serving).

**Ratios we'll create**:
- sugar_carb_ratio = sugars / carbohydrates (high = primarily sugar, bad)
- fat_energy_density = fat / energy (fat has 9 kcal/g)
- protein_density = proteins / energy (higher is better)
- sodium_mg = salt * 40 (convert to sodium for guidelines)

### Extract Binary Flags from Tags

Spencer's idea: we have rich info in tags but they're stored as messy JSON arrays. Let's extract the actual signals.

**From labels_tags**:
- is_vegetarian, is_vegan, is_organic, is_no_gluten, is_non_gmo

**From allergens_tags**:
- contains_nuts, contains_dairy, contains_gluten, contains_eggs

**From categories_tags**:
- is_beverage, is_snack, is_sweet

**From additives**:
- contains_additives (True if additives_n > 0)

Why does this matter? Instead of encoding thousands of tag combinations we get like 15-20 binary signals that actually predict grade. Organic, vegan, additive-free foods score differently - those are the signals we care about really.

### Simplify Categories

We have 10,000+ unique category combinations. Way too many for modeling.

Extract primary category, group into 8 tiers:
- Beverages, Snacks, Condiments, Dairy, Meat & Seafood, Bakery, Grocery, Other/Undefined

### Processing Indicators

More ingredients plus additives means more processed which usually means worse grades.

- processing_score = additives_n + ingredients_n
- palm_oil_indicator = Extract from ingredients_analysis_tags (True if contains "palm-oil" or "may-contain-palm-oil")


## Phase 2.1: Handle Highly Correlated Variables

From the correlation analysis we found some variables that move together:
- Energy and carbohydrates: 0.91 (very strong)
- Fat and energy: 0.89 (very strong)
- Sugars and added_sugars: 0.79 (strong)
- Proteins and energy: 0.69 (moderate-strong)
- Additives_n and ingredients_n: 0.68 (moderate-strong)

These make sense nutritionally - like fat and carbs both contribute to energy, or added sugars are part of total sugars. But when two variables are highly correlated they basically tell the model the same story twice, which can confuse some algorithms.

**Multicollinearity** when features are highly correlated (move together), some models struggle to figure out which one actually matters. Like if energy always goes up when carbs go up, the model can't tell if energy OR carbs is predicting the grade, or if it's both. Some models handle this fine, others don't.

**What we'll do**:
1. Keep energy, fat, carbs, sugars - they're related but not redundant (they measure different things)
2. Drop added_sugars - already 94% missing anyway, it's the same as sugars
3. Keep additives_n and ingredients_n - they mean different things (additives are chemical additives, ingredients is just the ingredient count)

If our models struggle (unstable predictions or irregular coefficients), we can:
- Use Ridge or Lasso regression which automatically handles correlations
- Apply PCA to smoosh correlated features together into new combined features
- Drop one from each pair (like keep energy, drop carbs)

For the baseline we'll keep most of them and hope the models can figure it out.


## Phase 3: Encode & Scale

### Scaling

We'll use RobustScaler (uses median and IQR, good at handling outliers). Energy might be 1,000 while proteins are 5 - without scaling the model thinks energy matters more just because the numbers are bigger, even though they're measuring different things.

Apply scaling to all nutrients and ratios, but NOT to completeness since it's already 0-1.

### Encoding

Challenge is we have hundreds of brands and thousands of tag combinations. One-hot encoding would create like 10,000+ columns which is way too much.

Strategy:
- Brands: Target encoding (take the mean nutriscore per brand) or just group into tiers (top 10 brands, top 100, everyone else)
- Categories: One-hot encoding (we reduced to 8 categories so this is doable now)
- Tags: Already binary flags from Phase 2, no encoding needed here
- Target (nutriscore_grade): Ordinal encoding (A=0, B=1, C=2, D=3, E=4)

Text features: Skip ingredients for the baseline, might add TF-IDF later if we need it.


## Phase 4: Handle Class Imbalance

### The Problem

Distribution is heavily skewed:
- E: 134,562 (32%)
- D: 98,591 (23%)
- C: 81,611 (19%)
- B: 47,033 (11%)
- A: 62,500 (15%)

Without balancing the model will just predict E and D all the time since those have the most examples.

### Our Approach

Start simple: Use `class_weight='balanced'` in models (LogisticRegression, RandomForest, XGBoost all support this). This makes the model pay more attention to rare classes and less to common ones.

If that's not enough add SMOTE to create synthetic samples for A and B but ONLY on training set. Validation and test stay original for unbiased evaluation.

If still struggling maybe combine undersample E+D with SMOTE A+B but gotta be careful not to overfit to synthetic samples.


## Phase 5: Validation Strategy (DO THIS BEFORE PHASE 4)

### Data Splits

- Train: 70%
- Validation: 15% (for hyperparameter tuning)
- Test: 15% (final unbiased evaluation)

All splits stratified to maintain class distribution.

### Cross-Validation

Use 5-fold stratified CV during hyperparameter tuning (GridSearch or RandomSearch) on training set only. Never ever tune hyperparameters on the test set or we'll fool ourselves.

### Evaluation Metrics

Primary: Macro F1-score - averages across all classes and accounts for imbalance

Secondary: Per-class precision/recall/F1, confusion matrix to see which classes we're screwing up, balanced accuracy


## Task Division

### Dev1 + Dev2: Data Cleaning
- Remove duplicates (brand+product, highest completeness)
- Cap outliers
- Standardize energy units (kJ→kcal)
- Handle missing values per strategy
- Deliverable: food_cleaned.csv

### Dev2: Feature Engineering
- Calculate nutrient ratios
- Extract binary flags from tags
- Simplify categories
- Processing indicators
- Deliverable: food_features.csv

### Dev3: Encoding & Scaling
- Implement RobustScaler
- Target encoding for brands
- One-hot for categories
- Ordinal encode target
- Deliverable: food_encoded.csv

### All Together: Split & Balance
- Stratified splits (70/15/15)
- Apply SMOTE to training set only
- Final validation
- Deliverable: train_processed.csv, val_processed.csv, test_processed.csv


## Success Criteria

We'll know we did it right when:
- No duplicates left in the data
- No extreme outliers (energy under 2000, carbs under 150, fat under 100)
- No missing values anywhere
- All energy in kcal (no 3000+ values that should be in kJ)
- Binary tag features created (around 15-20 new columns)
- All features encoded and scaled properly
- Train/val/test sets stratified by grade
- Processing finishes in under 60 minutes
- Datasets actually save and load without errors


## What Comes Next

After preprocessing, we'll:
1. Build baseline models (Logistic Regression, Random Forest)
2. Try advanced models (XGBoost, LightGBM)
3. Tune hyperparameters (validation set, 5-fold CV)
4. Evaluate on test set (macro F1, per-class metrics, confusion matrix)
5. Iterate if needed (add SMOTE, try ensemble methods)
6. Deliver: trained model, performance report, feature importance, recommendations

Goal is to understand which nutrients and categories predict Nutri-Score grades, so we can build a model that actually works.

