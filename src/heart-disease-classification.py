# %%
# Importing required libraries

# 📊 Exploratory Data Analysis (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# %%
# Load the dataset
df = pd.read_csv("../data/heart.csv")

# %%
# Display first 5 rows
print("🔹 First 5 Rows of Data:")
display(df.head())

# %%
# Basic info
print("🔹 Dataset Info:")
df.info()

# %%
# Summary statistics
print("🔹 Summary Statistics:")
display(df.describe())

# %%
# Check for missing values
print("🔹 Missing Values:")
print(df.isnull().sum())

# %%
# Target class distribution
print("🔹 Target Variable Distribution:")
sns.countplot(data=df, x='target')
plt.title('Heart Disease Presence (1) vs Absence (0)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# %%
# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# %%
# Distribution of numerical features
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# %%
# Count plots of categorical features
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, hue='target')
    plt.title(f"{col} vs Target")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Heart Disease")
    plt.show()

# %% [markdown]
# ## 🧹 Data Preprocessing

# %% [markdown]
# ### 🔧 Handling Missing Values

# %%
# No missing values in this dataset
# df.fillna(method='ffill', inplace=True)

# %% [markdown]
# ### 📐 Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ### 🔢 Encoding Categorical Variables

# %%
# All features are numeric in this dataset
# pd.get_dummies(df) or LabelEncoder

# %% [markdown]
# ## 📌 Feature Selection

# %%
# Correlation and domain knowledge used for feature selection
selected_features = X.columns.tolist()

# %% [markdown]
# ## 🧠 Model Building

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# %% [markdown]
# ### 🔹 Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# %% [markdown]
# ### 🔹 Decision Tree

# %%

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# %% [markdown]
# ### 🔹 Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# %% [markdown]
# ### 🔹 K-Nearest Neighbors

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# %% [markdown]
# ### 🔹 Support Vector Machine

# %%

from sklearn.svm import SVC

svm = SVC(probability=True)
svm.fit(X_train, y_train)

# %% [markdown]
# ## 📊 Model Evaluation

# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

models = {'Logistic Regression': lr, 'Decision Tree': dt, 'Random Forest': rf, 'KNN': knn, 'SVM': svm}

for name, model in models.items():
    print(f"📌 {name}")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print("="*50)


# %%

from sklearn.metrics import roc_curve

plt.figure(figsize=(8,6))

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=name)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %% [markdown]
# ## 🛠️ Hyperparameter Tuning (Example: Random Forest)

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)


