# Titanic Explorative Data Analysis

Perform exploratory data analysis and basic preprocessing on the Titanic dataset, including handling missing values, encoding categorical variables, standardizing numerical features, and detecting/removing outliers.

## Dataset
- **Titanic-Dataset.csv**

## Steps Performed
1. Loaded the dataset and explored structure, data types, and missing values.
2. Handled missing values:
   - Filled missing `Age` values using the median.
   - Filled missing `Embarked` values using the mode.
   - Dropped the `Cabin` column due to excessive missing data.
3. Converted categorical features (`Sex`, `Embarked`) into numerical features using one-hot encoding.
4. Standardized numerical features (`Age`, `Fare`, `SibSp`, `Parch`) using `StandardScaler`.
5. Visualized outliers using boxplots and removed outliers using the IQR method.

## Outputs
- Boxplots for numerical features are saved in the `outputs/` folder.
- The script prints dataset summaries and confirms preprocessing steps in the terminal.

## Tools Used
- Python
- pandas
- seaborn
- matplotlib
- scikit-learn

## What I Learned
- How to Create an enviroment for your project and why
- How to inspect a dataset to understand its structure, data types, and missing values before preprocessing.
- The importance of choosing appropriate strategies for handling missing data (median for numerical features, mode for categorical features).
- How to convert categorical variables into numerical form so they can be used in machine learning models.
- Why standardization is important for numerical features and how it ensures features are on a comparable scale.
- How boxplots help identify outliers and how the IQR method can be used to remove extreme values responsibly.
- The importance of performing preprocessing steps in a logical order to maintain data integrity.

## References
- ChatGPT
- https://www.youtube.com/watch?v=bDhvCp3_lYw
- https://www.youtube.com/watch?v=GP-2634exqA

## How to Run
```bash
pip install -r requirements.txt
python dataAnalysis.py
