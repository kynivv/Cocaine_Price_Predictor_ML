import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score as evs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


# Dataset Import
df = pd.read_csv('dream_market_cocaine_listings.csv')


# EDA & Preprocessing
print(df.info())

features = ['ships_from_to', 'grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'escrow', 'rating']

df = df[features]

print(df.info())

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

print(df.info())


# Train Test Split
target = df['btc_price']
features = df.drop('btc_price', axis= 1)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, shuffle= True, random_state= 42, test_size= 0.25)


# Model Training
models = [RandomForestRegressor(),
          GradientBoostingRegressor(),
          AdaBoostRegressor(),
          XGBRegressor(),
          DecisionTreeRegressor()
          ]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuray : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy : {evs(Y_test, pred_test)}\n')

    