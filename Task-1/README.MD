
# Titanic Data Cleaning & Preprocessing

## Project Overview
This task focuses on **cleaning, preprocessing, and preparing the Titanic dataset** for basic machine learning workflows.  
The goal is to transform raw data into a structured, numerical format suitable for analysis and modeling.

---

## Dataset
- **Input file:** `Titanic-Dataset.csv`
- **Output file:** `Titanic-Cleaned.csv`

The dataset contains passenger information such as age, sex, fare, embarkation point, and survival status.


## Libraries Used
  - `pandas` – data loading and manipulation  
  - `numpy` – numerical operations  
  - `matplotlib` & `seaborn` – data visualization  
  - `scikit-learn` – feature scaling  

## Steps Performed
  1. Data Loading & Exploration
    - Loaded the dataset using `pandas`
    - Displayed:
    - First 5 rows
    - Column information and data types
    - Missing values per column

  2. Handling Missing Values
    - Filled missing **Age** values using the **mean**
    - Filled missing **Embarked** values using the **most frequent category**
    - Dropped the **Cabin** column due to excessive missing values

  3. Dropping Non-Useful Columns
    - Removed text-based columns that are not suitable for basic ML models:
        - `Name`
        - `Ticket`

  4. Encoding Categorical Variables
    - Converted categorical features into numerical form using "one-hot encoding":
        - `Sex`
        - `Embarked`
  5. Feature Scaling
    - Standardized numerical features using **StandardScaler**:
        - `Age`
        - `Fare`
  6. Outlier Visualization
    - Visualized **Fare** outliers using a **boxplot**
  7. Saving Cleaned Data
    - Exported the final cleaned dataset as: `Titanic-Cleaned.csv`

## Refrences 
  - ChatGPT was the main guide for this project.
  - https://www.youtube.com/watch?v=GP-2634exqA

## How to Run
  1. Place `Titanic-Dataset.csv` in the same directory as the script  
  2. Install required libraries:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn
  3. Performed Cleaning operation using commands via VSCode 
