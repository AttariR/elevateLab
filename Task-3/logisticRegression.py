import os 
import numpy as np                      # for math and arrays
import pandas as pd                     # for reading CSV and handling tables
import matplotlib.pyplot as plt         # for plotting graphs
from sklearn.model_selection import train_test_split   # for splitting data
from sklearn.preprocessing import StandardScaler        # for standardizing features
from sklearn.linear_model import LogisticRegression     # the model we train
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve  # tools to evaluate model

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

## 1.Choose a binary classification dataset.
DATA_PATH = "Breast Cancer Wisconsin Dataset..csv"  
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Put it in this folder or fix the name/path.")

df = pd.read_csv(DATA_PATH)

if "diagnosis" in df.columns:
    y = df["diagnosis"]
    X = df.drop(columns=["diagnosis"])  
    y = y.map({"M": 1, "B": 0})
else:
    if "target" not in df.columns:
        raise KeyError("No 'diagnosis' or 'target' column found in the dataset.")
    y = df["target"]
    X = df.drop(columns=["target"])

for col in ["id", "ID", "Id", "Unnamed: 32"]:
    if col in X.columns:
        X = X.drop(columns=[col])

print("Shape:", df.shape)                          
print(df.head())  
print("\nFeatures shape:", X.shape)
print("Target values counts:\n", y.value_counts())

## 2.Train/test split and standardize features.
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2,
    random_state=42,
    stratify=y 
)

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   

## 3.Fit a Logistic Regression model.
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_scaled, y_train) 

y_pred = model.predict(X_test_scaled) 
y_proba = model.predict_proba(X_test_scaled)[:, 1]

## 4.Evaluate with confusion matrix, precision, recall, ROC-AUC.
cm = confusion_matrix(y_test, y_pred)         
precision = precision_score(y_test, y_pred)   
recall = recall_score(y_test, y_pred)         
roc_auc = roc_auc_score(y_test, y_proba)

output_file = open(f"{OUTPUT_DIR}/results.txt", "w")
print("\n--- Evaluation threshold = 0.5 ---")
print("Confusion Matrix:\n", cm, file=output_file)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)  
plt.figure()
plt.plot(fpr, tpr)                                
plt.plot([0, 1], [0, 1], linestyle="--")           
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (threshold = 0.5)")
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=200, bbox_inches="tight")
plt.close()

## 5.Tune threshold and explain sigmoid function.
custom_threshold = 0.3
y_pred_custom = (y_proba >= custom_threshold).astype(int)
# (y_proba >= threshold) returns True/False, then we convert to 1/0

cm2 = confusion_matrix(y_test, y_pred_custom)
precision2 = precision_score(y_test, y_pred_custom)
recall2 = recall_score(y_test, y_pred_custom)

print("\n--- Tuned threshold = 0.3 ---")
print("Confusion Matrix:\n", cm2, file=output_file)
print("Precision:", precision2)
print("Recall:", recall2)
print("ROC-AUC:", roc_auc)

z = np.linspace(-10, 10, 300)           
sigmoid = 1 / (1 + np.exp(-z))           

plt.figure()
plt.plot(z, sigmoid)
plt.xlabel("z (model score)")
plt.ylabel("sigmoid(z) = probability")
plt.title("Sigmoid Function")
plt.savefig(f"{OUTPUT_DIR}/sigmoid_curve.png", dpi=200, bbox_inches="tight")
plt.close()

output_file.close()