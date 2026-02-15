import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load data
df = pd.read_csv("Titanic-Dataset.csv")

print("FIRST 5 ROWS:")
print(df.head())

print("\nINFO (columns + types):")
print(df.info())

print("\nMISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# Fill Age with mean (average)
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df = df.drop(columns=["Cabin"])

print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

# 3) Drop text columns that aren't useful for basic ML models
df = df.drop(columns=["Name", "Ticket"])

# 4) Encode categorical columns (words -> numbers)
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

print("\nCOLUMNS AFTER ENCODING:")
print(df.columns)
print(df.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\nAFTER SCALING (Age, Fare):")
print(df[["Age", "Fare"]].head())

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df["Fare"])
plt.title("Fare Outliers (Boxplot)")
plt.show()


df.to_csv("Titanic-Cleaned.csv", index=False)
print("\nSaved: Titanic-Cleaned.csv")




