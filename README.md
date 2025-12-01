# 🥗 Open Food Nutrition Score Model
UCSD DSE 220 – Final Project (Fall 2025)  
Team: Spencer Hoyle, Francisco Chavezosa, Sean He


## Table of Contents
1. [Introduction](#1-introduction)
2. [Figures](#2-figures)
3. [Methods](#3-methods)  
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Conclusion](#6-conclusion)
7. [Statement of Collaboration](#7-statement-of-collaboration)
8. [Environment Setup](#8-environment-setup)

---

## 1. Introduction
The objective of this project is to develop a supervised machine learning model for predicting the **Nutri-Score (A–E)**, a key indicator of nutritional quality, for a wide range of food products.

![alt text](images/nutriscore.svg)<br>
[Learn more about Nutri-Score](https://world.openfoodfacts.org/nutriscore)

Utilizing the comprehensive **Open Food Facts** dataset — containing detailed nutritional and ingredient information for more than four million products worldwide — this study performs data preprocessing and feature engineering by extracting attributes from three primary domains:

- Nutrient values  
- Ingredient composition  
- Additive content

These processed features are used to train classification algorithms designed to accurately assign Nutri-Scores to new or unlabeled products.

![alt text](images/openfoodfacts.jpeg)
[Learn more about Open Food Facts](https://world.openfoodfacts.org/discover)


By automating the nutritional assessment process, the project seeks to:
- Demonstrate the potential of data-driven approaches in food health evaluation  
- Provide insights into the relative importance of nutritional and compositional factors that influence Nutri-Score classification  

The findings of this study aim to support consumers, manufacturers, and public health stakeholders in making more informed decisions about food quality and nutritional healthfulness.

---

## 2. Figures

This section presents selected figures used throughout our exploratory analysis and modeling workflow. Full-resolution versions of all figures can be found in the `plots/` directory.


-  [Figure 1: Summary Statistics](#figure-1-summary-statistics)
 -  [Figure 2: Duplicate Records](#figure-2-duplicate-records)
 -  [Figure 3: Top 10 Category Tags](#figure-3-top-10-category-tags)
 -  [Figure 4: Top 10 Food Group Tags](#figure-4-top-10-food-group-tags)
 -  [Figure 5: Top 10 Labels Tags](#figure-5-top-10-labels-tags)
 -  [Figure 6: Top 10 Additives Tags](#figure-6-top-10-additives-tags)
 -  [Figure 7: Top 10 Allergens Tags](#figure-7-top-10-allergens-tags)
 -  [Figure 8: Nutri-score Distribution](#figure-8-nutri-score-distribution)
 -  [Figure 9: Pearson Correlation Heatmap](#figure-9-pearson-correlation-heatmap)
 -  [Figure 10: Spearman Correlation Heatmap](#figure-10-spearman-correlation-heatmap)
 -  [Figure 11: Nutrient Feature Correlation Plot](#figure-11-nutrient-feature-correlation-plot)
 -  [Figure 12: Class SMOTE Balance](#figure-12-class-smote-balance)

## 3. Methods
This section summarizes the workflow used to extract, explore, preprocess, and prepare the data for modeling. Each step references the corresponding Jupyter notebook in the repository.

### 3.1 Data Extraction  
Notebook: [1_data_extraction.ipynb](notebooks/1_data_extraction.ipynb)

We extracted the Open Food Facts dataset from Hugging Face to be used in our project.

- Downloaded raw 4M+ row Open Food Facts dataset from Hugging Face ([food.parquet](https://huggingface.co/datasets/openfoodfacts/product-database/blob/main/food.parquet))
- Inspected structure, datatypes, and memory footprint  
- Selected essential columns and filtered incomplete records  
- Parsed JSON fields into tabular form  
- Exported cleaned dataset as `food.csv`

---

### 3.2 Data Exploration  
Notebook: [2_eda.ipynb](notebooks/2_eda.ipynb)

We analyzed the dataset to understand feature distributions, identify data quality issues, and explore relationships between nutritional variables.
- Reviewed dataset structure 
- Examined nutrient distributions and summary statistics  
- Analyzed missingness and duplicate records
- Identified outliers or anomolies
- Visualized Nutri-Score distribution  
- Identified strong correlations among nutrient features

---

### 3.3 Data Preprocessing  
Notebook: [3_data_preprocessing.ipynb](notebooks/3_data_preprocessing.ipynb)  
Documentation: [Preprocessing_Deliverable.md](documentation/Preprocessing_Deliverable.md)

We prepared the data for modeling through cleaning, feature engineering, encoding, scaling, and class balancing.

- **Clean Data**: Remove duplicates, cap outliers, fix units, handle missing values
- **Feature Engineering**: Calculate ratios, extract binary flags, simplify categories
- **Encode & Scale**: RobustScaler, target encoding, one-hot encoding
- **Split Data**: Stratified 70/15/15 train/val/test split
- **Balance Classes**: Apply SMOTE to training set only

**Important**: We split BEFORE balancing to prevent synthetic samples from leaking into validation/test sets.

### 3.4 First Model  
Notebook: [`notebooks/4_first_model.ipynb`](notebooks/4_first_model.ipynb)

This notebook establishes our baseline predictive model using the processed dataset, exploring initial classification performance.
- Loaded processed train/val/test splits  
- Trained baseline classification models  
- Evaluated early performance (accuracy, precision/recall, F1)  
- Generated initial confusion matrices for comparison with future models  

### 3.5 Second Model
Notebook: [`5_second_model.ipynb`](notebooks/5_second_model.ipynb)

---

## 4. Results
This section summarizes results of each method step.

### 4.1 Data Extraction (Results)
- Downloaded raw Dataset saved to `food.parquet` (4m rows)
- Filtered dataset for english products with valid nutriscore, and products with ingredenients, reducing size from 4m rows to 424k rows.

``` python
# Filter out invalid or blank nutriscore grade
df = df[df["nutriscore_grade"].isin(["a", "b", "c", "d", "e"])]

# Filter out products with no ingredients
df = df[~df["ingredients"].isna()]

# Filter for English language products
df = df[df["lang"] == "en"]
```

- Parsed Tags, Ingredients, and Nutriments and selected relevant features, reducing size from 110 columns to 24 columns.

``` python
final_cols = [
    "code", "brands", "product", "lang", "categories_tags", "food_groups_tags", "labels_tags",
    "additives_n", "additives_tags", "allergens_tags",
    "ingredients_analysis_tags", "ingredients_n", "ingredients_from_palm_oil_n", "ingredients",
    "completeness", "energy", "sugars", "added_sugars", "carbohydrates", "salt", "fat",
    "trans_fat", "proteins", "nutriscore_grade"
]
```
- Initally Processed Dataset saved to `food.csv` (424k rows)

### 4.2 Data Exploration (Results)

The dataset contains 424,297 observations and 24 features:
- `code`: Unique product code
- `brands`: Brand name of the food product
- `product`: Name of the food product
- `lang`: Language of the product
- `categories_tags`: List of category tags
- `food_groups_tags`: List of food group tags
- `labels_tags`: List of label tags
- `additives_n`: Number of additives
- `additives_tags`: List of additive tags
- `allergens_tags`: List of allergen tags
- `ingredients_ananlysis_tags`: List of ingredients analysis tags
- `ingredients_n`: Number of ingredients
- `ingredients_from_palm_oil_n`: Number of ingredients from palm oil
- `ingredients`: List of ingredients
- `completeness`: Completeness of product data (%)
- `energy`: Energy per 100g
- `sugars`: Sugar per 100g
- `added_sugar`: Added sugar per 100g
- `carbohydrates`: Carbohydrates per 100g
- `salt`: Salt per 100g
- `fat`: Fat per 100
- `trans_fat`: Trans fat per 100g
- `proteins`: Protein per 100g
- **`nutriscore_grade`**: Nutritional score grade (a,b,c,d,e)

---
#### Summary Statistics:
Numeric columns (nutrients and counts) are right-skewed, with many low values and a few extreme outliers.

- `energy` values vary widely, suggesting mixed units (kcal vs kJ) — will need normalization later.
- `completeness` averages around 0.6, meaning most records are moderately detailed.


##### **Figure 1: Summary Statistics**
![Summary Statistics](plots/summary_statistics.png)  

--- 
#### Duplicates:
We identified ~25,000 duplicate brand + product combinations. These may be true duplicates or distinct product variations.

##### **Figure 2: Duplicate Records**
![alt text](plots/duplicates.png)

---
#### Product Tags:
We identified the most frequently used tags in our product dataset to help understand the data and identify any areas for additional feature engineering.

##### **Figure 3: Top 10 Category Tags**
![Top 10 Category Tags](plots/top10_category_tags.png)

##### **Figure 4: Top 10 Food Groups Tags**
![Top 10 Food Groups Tags](plots/top10_food_groups_tags.png)

##### **Figure 5: Top 10 Labels Tags**
![Top 10 Labels Tags](plots/top10_labels_tags.png)

##### **Figure 6: Top 10 Additives Tags**
![Top 10 Additives Tags](plots/top10_additives_tags.png)

##### **Figure 7: Top 10 Allergens Tags**
![Top 10 Allergens Tags](plots/top10_allergens_tags.png)

---

#### Missing Values:
- added_sugars: 94% missing
- labels_tags: 72% missing
- trans_fat: 41% missing
- brands: 28% missing
- ingredients_from_palm_oil_n: 20% missing

---

#### Outliers:
We identified some outliers with certain products having significantly large amounts of `energy`, `sugars`, `carbohydrates`, `salt`, and `fat`.

---

#### Nutri-Score:
We analyzed the proportions of nutri-scores.

##### Figure 8: Nutri-score Distribution
![alt text](plots/nutri_score_pie_chart.png)

---

**Feature Correlation Results**

##### **Figure 9: Pearson Correlation Heatmap**  
Shows linear relationships between nutritional variables.  
![Pearson Correlation Heatmap](plots/pearson_correlation_heatmap.png)

##### **Figure 10: Spearman Correlation Heatmap**  
Captures monotonic (nonlinear) relationships among nutrients.  
![Spearman Correlation Heatmap](plots/spearman_correlation_heatmap.png)

##### **Figure 11: Nutrient Feature Correlation Plot**  
Visualizes pairwise nutrient interactions and their density distributions.  
![Correlated Nutritional Features](plots/correlated_nutritional_features.png)


---

### 4.3 Data Preprocessing (Results)

#### Phase 1: Clean the Data
- Removed duplicate brand/products and kept the record with the highested `completeness` (101,798 rows removed)
- Fixed mixed units (energy), converting kJ to kcal forv alues over 500.
- Capped outlier extrems at 99th percential for nutrient features.
- Handled missing values:
  - **added_sugars**: Drop
  - **labels_tags**: Replace null with [] (empty list)
  - **brands**: Fill with "unknown"
  - **product**: Drop
  - **ingredients_from_palm_oil_n**: Drop
  - **Main nutrients**: Fill with median
  - **ingredient_n**: Drop
- Results saved to `food_clean.csv` (346,071 rows, 22 columns)

#### Phase 2: Feature Engineering
- Calculate Nutrient Ratios
  - **sugar_carb_ratio**: sugars / carbohydrates (high = primarily sugar)
  - **fat_energy_density**: fat / energy (fat has 9 kcal/g)
  - **protein_density**: proteins / energy (higher is better)
  - **sodium_mg**: salt * 400 (convert to sodium for guidelines)
- Extract Binary Flags from Tags (`is_vegatarian`, `is_vegan`, `is_organic`, etc)
<br><br>![alt text](plots/binary_tag_flags.png)<br>
- Simplify Categories, reducing 10,000 + category combinations to 8 primary tiers
  - Beverages: 103152
  - Snacks: 84670
  - Other/Undefined: 81992
  - Condiments: 26245
  - Dairy: 23719
  - Meat & Seafood: 21600
  - Grocery: 4262
  - Bakery: 431
- Processing Indicators

#### Phase 3: Encode & Scale
- Scale numeric features using RobustScaler (median and IQR)
- Encode Categorical Features
- Encode Target Varaibles (ordinal)
  - A = 0
  - B = 1
  - C = 2
  - D = 3
  - E = 4
- Split data into 70% train, 15% validation, 15% test
- SMOTE successfully balanced minority Nutri-Score classes in the training set

##### Figure 12: Class SMOTE Balance
![alt text](plots/SMOTE.png)

- Processed Data saved
  - `train_processed.csv`: (384,395 rows, 38 columns)
  - `val_processed.csv`: (51,911 rows, 38 columns)
  - `test_processed.csv`: (51,911 rows, 38 columns)

### 4.4 First Model (Results)

### 4.5 Second Model (Results)

---

## 5. Discussion

This section summarizes the reasoning behind our methodological choices, interprets the results, and highlights limitations in the project.

### 5.1 Data Extraction
Our extraction and filtering choices were driven by the need to create a reliable, consistent subset of the Open Food Facts dataset. Restricting the data to English-language products, valid Nutri-Score entries, and products with ingredient lists ensured that downstream models were trained on clean and usable information.

### 5.2 Data Exploration
EDA revealed several challenges—highly skewed nutritional distributions, missing values in key fields, and a strong imbalance across Nutri-Score grades—which directly informed our preprocessing steps.

Nutri-Score analysis:
- The blue/green slices (d, e) dominate the dataset.
- Healthy classes a and b occupy smaller areas.
- This imbalance might affect model learning and should be taken into consideration.

**1. Strong Positive Correlations**
These reflect expected nutritional relationships:
- **Energy ↔ Carbohydrates (0.91)** — Carbs are a major calorie source.  
- **Energy ↔ Fat (0.89)** — Fat contributes the most calories per gram (9 kcal).  
- **Energy ↔ Proteins (0.69)** — Protein adds calories but less than fat/carbs.  
- **Carbohydrates ↔ Sugars (0.65)** — Sugars are a subset of total carbs.  
- **Ingredients_n ↔ Additives_n (0.68–0.72)** — More complex/processed foods contain more additives.

**2. Moderate or Weak Correlations**
- **Added sugars ↔ Sugars (0.71)** — Added sugars contribute significantly to total sugars.  
- **Added sugars ↔ Carbohydrates (0.42)** — High added-sugar foods trend toward high carbs.  
- **Energy ↔ Sugars (moderate)** — Sugary products contribute calories but not as strongly as fats or complex carbs.  
- **Salt ↔ Other nutrients (near 0)** — Salty and sweet foods represent distinct product categories.  

**3. Negative or Near-Zero Correlations**
- **Proteins ↔ Sugars (–0.22)** — High-protein foods tend to have lower sugar content.  
- **Completeness ↔ All nutrients (≈0)** — Completeness reflects data quality, not nutrition.  
- **Trans_fat ↔ Nutrients (≈0)** — Trans fat levels relate more to regulation than general nutrition.

**Insights From Pairwise Nutrient Plots**
- **Energy vs. Fat** shows the strongest upward trend; fat-rich foods are the most calorie-dense.  
- **Energy vs. Carbohydrates** is positive but more variable due to different product types.  
- **Energy vs. Sugars** shows moderate correlation—sugar contributes calories, but not as dominantly as fats.  
- **Energy vs. Protein** has a mild positive slope, consistent with protein’s lower caloric density.

### 5.2 Preprocessing Decisions
The preprocessing pipeline was designed to correct inconsistencies and create a modeling-ready dataset.  
- **Outlier capping** and **unit normalization** (especially for energy) were necessary to prevent extreme values from distorting model learning.  
- **Feature engineering** clarified nutritional relationships by introducing meaningful ratios and dietary indicators.  
- **Encoding and scaling** ensured numeric stability and allowed categorical features (e.g., tags) to contribute meaningfully to prediction.  
- **SMOTE** was applied to address class imbalance, which was crucial for preventing the model from collapsing toward predicting majority classes (C, D, and E).

These decisions collectively improved the quality and structure of the inputs fed to the models.

### 5.3 Model Interpretation


### 5.4 Believability of Results & Shortcomings
The results appear scientifically consistent: strong correlations among nutrients aligned with nutritional principles, and the model favored features known to drive Nutri-Score (fat, sugar, salt, energy density). However, several limitations remain:

- **Data Quality:** Many fields were incomplete, inconsistent, or self-reported, which introduces noise into both features and labels.
- **Category & Ingredient Variability:** Tags and ingredient lists are highly unstructured, limiting their usefulness without more advanced NLP processing.
- **Nutri-Score Itself:** The scoring system is rule-based and nonlinear; machine learning can approximate it but cannot perfectly replicate edge cases.

### 5.5 Overall Reflection
The project successfully demonstrated that Nutri-Score can be predicted with reasonable accuracy using structured nutrient data and engineered features. The workflow is reproducible, scalable, and provides a strong foundation for more advanced modeling. Future improvements—such as ingredient-list NLP, product-image modeling, or hybrid rule-based + ML systems—could meaningfully increase performance.


---

## 6. Conclusion
This project successfully built a machine learning pipeline capable of predicting Nutri-Score classifications using structured nutritional data.  
Key outcomes:

- Demonstrated the value of feature engineering  
- Established a reproducible preprocessing pipeline  
- Built and evaluated a baseline predictive model  

**Future directions:**
- Incorporate NLP from ingredient text  
- Use image features from product photos  
- Experiment with deep learning architectures  
- Improve handling of missing nutrient data  

---

## 7. Statement of Collaboration

**Spencer Hoyle – Team Lead & Writing**  
Led data extraction, preprocessing pipeline, project structure, and final report writing.

**Sean He – EDA & Modeling**  
Performed exploratory data analysis, contributed to modeling and evaluation.

**Francisco Chavezosa – Feature Engineering & Modeling**  
Designed engineered features, contributed to modeling and evaluation.

---

## 8. Environment Setup

1. Clone the repository
```bash
git clone https://github.com/swhoyle/food-nutrition-model.git
cd open-food-nutrition-score
```

2. Install dependencies ([requirements.txt](requirements.txt))
```
pip install -r requirements.txt
```

3. Open and run the notebooks in the `notebooks/` folder

- [1_data_extraction.ipynb](notebooks/1_data_extraction.ipynb): Download and extract dataset (`food.parquet` → `food.csv`)  
- [2_eda.ipynb](notebooks/2_eda.ipynb): Exploratory data analysis
- [3_data_preprocessing.ipynb](notebooks/3_data_preprocessing.ipynb): Data cleaning and preprocessing
- [4_first_model.ipynb](notebooks/4_first_model.ipynb): Build our first prediction model
- [5_second_model.ipynb](notebooks/5_second_model.ipynb): Build our first prediction model