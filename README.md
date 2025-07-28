# heart_disease_classification
Heart Disease Classification: Predicting Cardiovascular Risk with Machine Learning

This project focuses on developing a machine learning model to classify the presence of heart disease based on patient health metrics. Cardiovascular disease remains one of the leading causes of death worldwide, and early prediction plays a crucial role in saving lives.

📌 Objective
To build and evaluate classification models that can accurately predict the likelihood of heart disease using structured health data, enabling better preventive healthcare decisions.

📊 Dataset
The dataset used in this project is based on the Cleveland Heart Disease dataset, one of the most commonly used datasets in medical machine learning.

# Features include:

- Age

- Sex

- Chest pain type (cp)

- Resting blood pressure (trestbps)

- Serum cholesterol (chol)

- Fasting blood sugar (fbs)

- Resting electrocardiographic results (restecg)

- Maximum heart rate achieved (thalach)

- Exercise-induced angina (exang)

- ST depression (oldpeak)

- Slope of the peak exercise ST segment (slope)

- Number of major vessels colored by fluoroscopy (ca)

- Thalassemia (thal)


** Target variable: **

- 0: No heart disease

- 1: Presence of heart disease


# 🧠 Machine Learning Workflow

1. Data Collection

2. Exploratory Data Analysis (EDA)

3. Data Preprocessing

    - Handling missing values

    - Feature scaling

    - Encoding categorical variables

4. Feature Selection

5. Model Building

    - Logistic Regression

    - Decision Tree

    - Random Forest

    - K-Nearest Neighbors (KNN)

    - Support Vector Machine (SVM)

6. Model Evaluation

    - Accuracy, Precision, Recall, F1-Score

    - ROC-AUC Curve

7. Hyperparameter Tuning

    - GridSearchCV, Cross-Validation

8. Deployment-ready Notebook


# 🔍 Key Insights
    - Importance of different features (like age, chest pain, and thalassemia) in predicting heart disease

    - Comparison of multiple ML algorithms

    - ROC curves and confusion matrices to understand model performance


# 📈 Results
The Random Forest and SVM classifiers achieved the best performance with:

    - Accuracy: ~85%

    - Precision and Recall: balanced across both classes

    - ROC-AUC: >0.90

# 🚀 Future Improvements
    - Integrate with real-time health monitoring systems

    - Use deep learning for more complex feature interactions

    - Deploy as a web application using Flask or Streamlit

# 🗂️ Folder Structure

heart-disease-prediction/
│
├── data/                    # Dataset files
├── notebooks/               # Jupyter Notebooks with code
├── models/                  # Saved model files (e.g. .pkl)
├── reports/                 # EDA and model performance reports
├── requirements.txt         # Python dependencies
└── README.md                # Project overview


# 📚 References
    - UCI Heart Disease Dataset

    - Research on cardiovascular risk factors and machine learning applications in healthcare

# 🙌 Contributions
Suggestions and improvements are welcome! Please fork the repo, create a pull request, or open an issue.