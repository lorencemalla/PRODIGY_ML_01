# PRODIGY_ML_01
Implementing a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

The primary goal of this code is to build a machine learning model to predict house prices based on various features from a dataset. The target variable is SalePrice.


Libraries Used:-
1. pandas, numpy: For data manipulation and numerical operations.
2. seaborn, matplotlib.pyplot: For data visualization (though not heavily used here).
3. scikit-learn: For preprocessing, model building, and evaluation.
4. train_test_split: Splits data into training and test sets.
5. LinearRegression: The regression model used.
6. StandardScaler: Standardizes numerical features.
7. SimpleImputer: Handles missing data.
8. OneHotEncoder: Encodes categorical variables.
9. SelectKBest: Selects the top K features based on statistical tests.
10. Pipeline and ColumnTransformer: Combine preprocessing and modeling steps.

Significance:-
Demonstrates end-to-end machine learning: from data cleaning to evaluation.
Uses scikit-learn’s pipeline architecture for cleaner and modular code.
Helps understand how to handle mixed-type features and missing data in real-world datasets.
