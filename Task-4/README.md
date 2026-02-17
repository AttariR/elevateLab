Task 4: Classification with Logistic Regression

1. Objective
Build a binary classification model using Logistic Regression, evaluate its performance using standard metrics, tune the decision threshold, and understand the sigmoid function.
This task is part of the AI & ML Internship – Task 4.

2. Tools and Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

3. Dataset
- Name: Breast Cancer Wisconsin Dataset
- Samples: 569
- Features: 30 numerical features
- Target Variable:
    - 0 → Benign
    - 1 → Malignant

4. Approach
## Data Preparation
    - The dataset was loaded using Pandas.
    - Non-feature columns such as id and Unnamed: 32 were removed.
    The target variable was encoded where M was mapped to 1 and B to 0.
    - Class distribution was checked to understand the balance of the dataset.

## Train-Test Split and Scaling
    - The data was split into 80 percent training data and 20 percent testing data.
    - Stratified sampling was used to preserve class balance.
    Features were standardized using StandardScaler.

## Model Training
    - A Logistic Regression model was trained using scikit-learn.
    - Both class predictions and probability outputs were generated.


5. What I Learned
- I learned how Logistic Regression performs binary classification.
- Understood the importance of feature scaling before training models, how ROC-AUC evaluates model performance independently of thresholds and learnt the difference between precision and recall and when each metric is important and 
- Gained a clearer understanding of the sigmoid function and how it converts model scores into probabilities while also improving my ability to debug common machine learning and Python errors.

6. References
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- ChatGPT (OpenAI) was used for guidance on debugging, evaluation metrics, and documentation.

7. Conclusion
This project demonstrates a complete Logistic Regression workflow, from data preparation to evaluation and threshold tuning, and fulfills all the requirements of Task 4.