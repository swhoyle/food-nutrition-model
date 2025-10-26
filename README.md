# 🥗 Open Food Nutrition Score Model

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
git clone https://github.com/yourusername/open-food-nutrition-score.git
cd open-food-nutrition-score
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Open `food_nutrition_model.ipynb` and run the cells.