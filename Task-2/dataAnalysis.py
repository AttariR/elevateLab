import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


sns.set(style="whitegrid")

DATA_PATH = "Titanic-Dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Ensure Titanic-Dataset.csv is in this folder.")

df = pd.read_csv(DATA_PATH)

print("Loaded Dataset")
print("Shape (rows, cols):", df.shape)
print(df.head())

print("\n--- INFO (columns, types, missing counts) ---")
print(df.info())

print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum().sort_values(ascending=False))

print("\n--- SUMMARY STATS (NUMERIC) ---")
print(df.describe())

##- 2. Missing Values
#Replace 177 missing "Age" values using the median 
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

#Replace 2 missing "Embarked" values using the mode 
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#Replace 687 missing "Cabin" values using the drop -- too many missing values, so removes 'Cabin' column
if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])

print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum().sort_values(ascending=False))

## 3. Encoding
os.makedirs("outputs", exist_ok=True)
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

print("\n--- COLUMNS AFTER ENCODING ---")
print(df.columns)

##- 4. Standardize Numerical Features 
scaler = StandardScaler()
num_cols = [c for c in ["Age", "Fare", "SibSp", "Parch"] if c in df.columns]

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\n--- FIRST 5 ROWS AFTER STANDARDIZATION ---")
print(df[num_cols].head())

print("\n--- CHECK MEAN & STD (should be ~0 mean, ~1 std) ---")
print(df[num_cols].agg(["mean", "std"]).round(3))

##- 5.Boxplots
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/box_{col}.png")
    plt.close()

##- Remove outliers using IQR method
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\n--- DATA SHAPE AFTER OUTLIER REMOVAL ---")
print(df.shape)

df.to_csv("outputs/titanic_cleaned.csv", index=False)
print("Saved cleaned dataset to outputs/titanic_cleaned.csv")
