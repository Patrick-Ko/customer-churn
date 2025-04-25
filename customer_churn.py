# Import librabries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


from sklearn.model_selection import train_test_split #split dataset
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler  #handle outliers
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression  #model to be trained
from sklearn.tree import DecisionTreeClassifier  #model to be trained
from sklearn.ensemble import RandomForestClassifier  #model to be trained
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay  #evaluation
from imblearn.over_sampling import SMOTE



import joblib 
import pickle

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("C:\\Users\\Komla\\Downloads\\Mobile Net. Telecom Analysis.csv")

# data.head()
# data.shape
# data.info()
#data.drop(columns=['Customer ID', 'Phone Number', 'State', 'Churn Category', 'Churn Reason'], inplace=True)
#data.head()
# Droppingcolumns
data_cleaned = data.drop(columns=['Customer ID', 'Phone Number', 'State', 'Churn Category', 'Churn Reason'])

# Converting target column to binary
data_cleaned['Churn Label'] = data_cleaned['Churn Label'].map({'Yes': 1, 'No': 0})

#encoding categorical variables
categorical_cols = data_cleaned.select_dtypes(include='object').columns
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_cols, drop_first=True)

#data.head()

# Features and target
X = data_encoded.drop('Churn Label', axis=1)
y =data_encoded['Churn Label']

# Save the feature names to a file for reference
feature_names = X.columns.tolist()
with open('C:\\Users\\Komla\\Downloads\\feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Using Logistic Regression Model (Base model)

# To Train the model
logreg = LogisticRegression(max_iter=1000, random_state=42)   
logreg.fit(X_train, y_train)

# Prediction
y_pred_logreg = logreg.predict(X_test)

# Classification Report
# print("Logistic Regression Report:")
# print(classification_report(y_test, y_pred_logreg))

conf_matrix = confusion_matrix(y_test,y_pred_logreg)

# Display the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logreg.classes_)
# disp.plot(cmap="Greens")
# plt.title("Confusion Matrix")
# plt.show()
#Using Random Forest Classifier Mod

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)


# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# Using Decision Tree Classifier Model
# Train model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Predict
y_pred_dtree = dtree.predict(X_test)

# Evaluate
# print("Decision Tree Report:")
# print(classification_report(y_test, y_pred_dtree))

conf_matrix = confusion_matrix(y_test,y_pred_dtree)

# Display the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
# disp.plot(cmap="Reds")
# plt.title("Confusion Matrix")
# plt.show()
## Save the model
joblib.dump(model, 'C:\\Users\\Komla\\Downloads\\random_forest_model.joblib')

# Save the scaler
#joblib.dump(scaler, 'scaler.pkl')






