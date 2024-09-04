# currently not used, not maybe be used in the future
# instead of this, preprocessing pipeline was exported directly from notebook and used in main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df_train = pd.read_csv("train.csv")

X_train = df_train.drop("SalePrice", axis = 1)
# y_train = np.log(df_train["SalePrice"])

# Define transformers for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

categorical_columns = pd.Index(['MSZoning', 'HouseStyle'])

numerical_columns = pd.Index(['LotArea', 'YearBuilt', 'TotRmsAbvGrd'])

df_cats = X_train[categorical_columns]

df_nums = X_train[numerical_columns]

X_train = pd.concat([df_nums, df_cats], axis=1)

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ],remainder = 'passthrough')

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)])

pipeline = pipeline.fit(X_train)