# Property Price Prediction Project
## Project Overview
This project aims to develop a robust machine learning model for accurately predicting property prices based on various features. The goal is to identify key factors influencing property values and build a model that generalizes well to unseen real estate data. The solution incorporates extensive data preprocessing, advanced feature engineering, and the evaluation of multiple machine learning models to achieve high predictive accuracy.
### Key Features
**Comprehensive Data Preprocessing:** Handling missing values, outlier treatment, and data standardization.

**Advanced Feature Engineering:** Creation of time-based features, area-based ratios, and combined room features to enhance model intelligence.

**New Data Integration:** A pipeline for integrating external datasets, ensuring consistency and augmenting training data.

**Multiple Model Evaluation:** Comparison of Linear Regression, Random Forest, XGBoost, and Stacking Regressor models.

**Hyperparameter Tuning:** Optimization of model parameters using GridSearchCV for enhanced performance.

**Robust Prediction Engine:** A function to make predictions for new, custom property inputs.

**Detailed Performance Analysis:** Evaluation using R-squared scores, feature importance analysis, and residual analysis.

## How to Run the Project
To run this project, you will need a Google Colab environment or a local Jupyter Notebook setup with Python and the necessary libraries installed.

**Prerequisites**
Ensure you have the following Python libraries installed:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn kagglehub

### Steps to Execute
Download the Notebook: Save this .ipynb file to your local machine.
Upload to Google Colab / Open in Jupyter: Upload the notebook to Google Colab or open it in your local Jupyter environment.

Data Files: Ensure the train_part1.csv and train_part2.csv files are uploaded to your Colab environment or are accessible in the same directory as your notebook if running locally. The additional Kaggle dataset (real_estate_main.csv) will be downloaded programmatically by the notebook.

Run All Cells: Execute all cells in the notebook sequentially. The notebook is designed to guide you through data loading, preprocessing, feature engineering, model training, evaluation, and interactive prediction.

### Key Results
**Best Performing Model:** The Stacking Regressor (combining Random Forest and XGBoost) achieved the highest predictive performance with a Test R-squared score of 0.9696. The tuned XGBoost model was a very close second with a Test R-squared of 0.9684.

**Feature Importance:** Key drivers of property prices were identified:

**Area Metrics:** BuiltUpArea_sqft (0.391) and SuperBuiltUpArea_sqft (0.099) were highly influential.

**Geographical Location:** City_MMR (0.306) and Longitude (0.026) were paramount.

**Property Type:** PropertyType_Penthouse (0.026) significantly impacted prices.

**Impact of Log Transformation:** Log-transforming the target variable (Price_INR) was crucial for handling its skewed distribution, leading to more stable model training and improved accuracy.

**Working Prototype:** The system can predict property prices based on user-defined inputs, demonstrating its practical utility.


### Links for datasets
[(https://www.kaggle.com/datasets/luv00679/delhi-ncr-real-estate-data); (https://drive.google.com/file/d/1esg4lels10ptwXupTfT1u7sm3FsDFxsm/view?usp=sharing); (https://drive.google.com/file/d/1i3v9jLjRGPtW0-KBkeY9SCkGrwsD03Qa/view?usp=sharing)]

real_estate_main.csv (https://www.kaggle.com/datasets/luv00679/delhi-ncr-real-estate-data)

train_part1.csv(https://drive.google.com/file/d/1esg4lels10ptwXupTfT1u7sm3FsDFxsm/view?usp=sharing)

train_part2.csv(https://drive.google.com/file/d/1i3v9jLjRGPtW0-KBkeY9SCkGrwsD03Qa/view?usp=sharing)


### Limitations

Outlier Prediction Challenges: The model still shows some discrepancies in predicting prices for ultra-premium and high-error properties, suggesting nuances not fully captured.

Reliance on Imputation: Heavy imputation for certain features in new data might sometimes dilute property uniqueness.

### Future Enhancements:

Enhanced feature engineering for luxury amenities, builder reputation, and specific location advantages.

Deeper investigation into high-error cases through manual data audits.

Exploration of advanced modeling techniques like Quantile Regression or Hierarchical Models.

Continuous model retraining with updated market data.

**Tools/Frameworks:** Python, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, Google Colab, kagglehub.
