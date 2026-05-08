import pandas as pd
import numpy as np

df = pd.read_csv('AmesHousing.csv')

print(df.shape)           # rows and columns
print(df.head())          # first 5 rows
print(df.dtypes)          # column data types
print(df['SalePrice'].describe())  # price statistics

missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title('House price distribution')
plt.xlabel('Sale Price ($)')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['SalePrice']), bins=50, kde=True)
plt.title('Log-transformed price')
plt.xlabel('Log(Sale Price)')

plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150)
plt.show()

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
print(corr.head(10))

plt.figure(figsize=(8, 6))
corr.head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 features correlated with sale price')
plt.tight_layout()
plt.savefig('correlations.png', dpi=150)
plt.show()

features = [
    'Overall Qual', 'Gr Liv Area', 'Garage Cars',
    'Total Bsmt SF', '1st Flr SF', 'Full Bath',
    'TotRms AbvGrd', 'Year Built', 'Year Remod/Add',
    'Fireplaces'
]

X = df[features].copy()
y = np.log1p(df['SalePrice'])

X.fillna(X.median(), inplace=True)

print(X.shape)
print(X.isnull().sum())

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'Linear Regression':      LinearRegression(),
    'Random Forest':          RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':      GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    results[name] = scores.mean()
    print(f"{name}: R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    best_name = max(results, key=results.get)
print(f"\nBest model: {best_name} (R² = {results[best_name]:.4f})")

best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', models[best_name])
])
best_pipe.fit(X_train, y_train)

joblib.dump(best_pipe, 'model.pkl')
joblib.dump(features, 'features.pkl')
print("Model saved!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred_log = best_pipe.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

mae  = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2   = r2_score(y_test, y_pred_log)

print(f"R² Score:  {r2:.4f}")
print(f"MAE:       ${mae:,.0f}")
print(f"RMSE:      ${rmse:,.0f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred, alpha=0.4, color='steelblue', s=20)
plt.plot([y_actual.min(), y_actual.max()],
         [y_actual.min(), y_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()

if hasattr(best_pipe['model'], 'feature_importances_'):
    importances = best_pipe['model'].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values()
    feat_imp.plot(kind='barh', color='steelblue', figsize=(8, 5))
    plt.title('Feature importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()