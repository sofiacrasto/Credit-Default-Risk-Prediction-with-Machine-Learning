# Credit-Default-Risk-Prediction-with-Machine-Learning

Predicting customer default risk with a powerful combination of data exploration, feature engineering, and multi-model comparison.

📌 Overview

This project presents a full end-to-end machine learning pipeline designed to predict whether a customer is likely to default on a credit obligation. Using structured financial and demographic data, the solution guides the reader from raw data analysis through model training, evaluation, and interpretation.

🧠 Objectives
- Understand the drivers of risky customer behavior using exploratory data analysis
- Engineer features and preprocess data for optimal model performance
- Compare multiple classification algorithms to identify top-performing models
- Use feature importance to explain model predictions
- Share actionable insights with visual storytelling and business context

📊 Exploratory Data Analysis
- Visualized missing values using missingno and heatmaps
- Countplots and histograms for demographic and financial distributions
- Boxplots and KDE plots for comparing defaulters vs. non-defaulters
- Correlation matrix to identify multicollinearity and driver features

🛠️ Data Processing
- Created a binary target column from Defaults Records
- Scaled numerical features using MinMaxScaler
- One-hot encoded all categorical variables (Gender, Marital Status, etc.)
- Final dataset composed of 21 features and a balanced target column

🤖 Modeling Pipeline
Trained and evaluated the following classifiers:
| Classifier | 
| Logistic Regression | 
| K-Nearest Neighbors | 
| Random Forest | 
| XGBoost | 
| Decision Tree | 
| SVC | 
| Extra Tree Classifier | 
| Gaussian NB | 
| SGD Classifier | 
| AdaBoost | 
| Gradient Boosting | 
| Bagging Classifier | 


Each model was evaluated using accuracy, precision, recall, and F1-score.

🏆 Model Performance Summary
- Top Performer: KNeighborsClassifier achieved an F1-score of ~0.87.
- XGBoost showed strong recall but slightly lower precision on a small test set.
- Feature Importance from RandomForestClassifier revealed that:
- Credit Score, Debt-to-Income Ratio, and Income were the most impactful predictors.
  
📈 Visuals:
- Bar chart comparing F1-scores across all models
- Feature importance plot to enhance explainability

📌 Business Impact
This model enables financial institutions to:
- Identify high-risk profiles before extending credit
- Tailor credit policies based on customer segmentation
- Allocate risk-adjusted resources more efficiently

📁 Repository Structure
risk-default-analysis/
│
├── risk_behavior_features.csv     # Raw dataset
├── risk_default_analysis.ipynb    # Jupyter notebook with full analysis
├── README.md                      # Project documentation



📌 Tools & Technologies
- Python, Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- scikit-learn, XGBoost
- missingno, MinMaxScaler, classification_report
