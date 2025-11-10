# 🥗 Open Food Nutrition Score Model

UCSD DSE 220 Project - Fall 2025<br>
Team: Spencer Hoyle, Francisco Chavezosa, Sean He

## 📘 Abstract
The objective of this project is to develop a **supervised machine learning model** for predicting the **Nutri-Score (A–E)**, a key indicator of nutritional quality, for a wide range of food products.  

Utilizing the comprehensive **Open Food Facts** dataset — containing detailed nutritional and ingredient information for more than **four million products worldwide** — this study performs **data preprocessing** and **feature engineering** by extracting attributes from three primary domains:
- Nutrient values  
- Ingredient composition  
- Additive content  

These processed features are used to train classification algorithms designed to accurately assign Nutri-Scores to new or unlabeled products.  

By automating the nutritional assessment process, the project seeks to:
- Demonstrate the potential of data-driven approaches in food health evaluation  
- Provide insights into the relative importance of nutritional and compositional factors that influence Nutri-Score classification  

The findings of this study aim to support **consumers**, **manufacturers**, and **public health stakeholders** in making more informed decisions about food quality and nutritional healthfulness.

---

## 📊 Dataset
- **Source:** [Open Food Facts](https://world.openfoodfacts.org/)  
- **Dataset File:** [food.parquet – Hugging Face Dataset](https://huggingface.co/datasets/openfoodfacts/product-database/blob/main/food.parquet)

---

## ⚙️ Environment Setup

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
- [2_eda.ipynb](notebooks/2_eda.ipynb): Exploratory data analysis using `food.csv`  
- [3_preprocessing.ipynb](notebooks/3_preprocessing.ipynb): Data cleaning and preprocessing (`food.csv` → `food_cleaned.csv`)  
- [4_model_building.ipynb](notebooks/4_model_building.ipynb): Build prediction model using `food_cleaned.csv`

# 1. Data Extraction

The first stage of this project involved obtaining and preparing the Open Food Facts dataset, a large open-source database containing detailed product information for food items from around the world. The dataset, available on Hugging Face as `food.parquet`, contains approximately 4 million rows and over 110 columns, including several semi-structured fields stored in JSON format.

After downloading the dataset, we performed an initial inspection to understand its dimensions, data types, and memory requirements, identifying which fields required transformation. We then applied filters to retain only entries meeting project-specific criteria, ensuring that products had sufficient and reliable data. Key attributes such as product identifiers, names, categories, ingredients, and nutritional metrics were selected, while unnecessary or low-quality fields were removed.

JSON-encoded columns were parsed and flattened into standard tabular form, and column names were reformatted for clarity and consistency. The resulting dataset was exported as `food.csv`, representing a structured and analysis-ready version of the Open Food Facts data. This refined subset provides a strong foundation for exploratory data analysis and subsequent preprocessing steps.

# 2. Exploratory Data Analysis

TThe next step is to perform Exploratory Data Analysis (EDA) on our dataset `food.csv`. We want to understand the data structure, distributions, and relationships between features, and to identify any data quality issues before preprocessing.

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



![alt text](plots/pearson_correlation_heatmap.png)

![alt text](plots/spearman_correlation_heatmap.png)

![alt text](plots/correlated_nutritional_features.png)

# 3. Data Preprocessing