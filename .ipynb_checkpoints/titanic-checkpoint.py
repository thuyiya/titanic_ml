
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE  # For handling data imbalance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from google.colab import drive
import missingno as msno

# :D no warinig pleasee....
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Mount Google Drive
drive.mount('/content/drive')

# Paths to the dataset files
train_data_path = "/content/drive/MyDrive/assesment_dataset/train.csv"
test_data_path = "/content/drive/MyDrive/assesment_dataset/test.csv"

# Load the training dataset
df = pd.read_csv(train_data_path)

# Exploratory Data Analysis (EDA)
print("Dataset Overview:\n", df.info())
print("\nFirst 5 Rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Visualization of Missing Data
msno.matrix(df)
plt.show()

# Visualizing survival rates across multiple features
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x='SibSp', y='Survived', data=df)
plt.title("Survival Rate by Family Members")
plt.show()

sns.barplot(x='Parch', y='Survived', data=df)
plt.title("Survival Rate by Parents/Children")
plt.show()

sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Survival Rate by Fare")
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Survival Rate by Age")
plt.show()

sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Survival Rate by Embarked Port")
plt.show()

# Feature Engineering
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Derive 'Title' from the 'Name' column
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                    'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Create bins for 'Age' and 'Fare'
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Mid', 'High', 'Very High'])

# Add 'FamilySize' and 'IsAlone' features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Add 'Fare_Per_Person' feature
df['Fare_Per_Person'] = df['Fare'] / (df['FamilySize'] + 1)

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin'], drop_first=True)

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Define features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Handle data imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and compare models with cross-validation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', C=1, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model_name = None
best_accuracy = 0

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Performance:")
    print("Cross-Validated Accuracy:", mean_cv_score)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_model_name = name

# Plot Confusion Matrix for the Best Model

# Train the best model on the full training set
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Make predictions
y_pred_best = best_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()

# Correlation Heatmap

# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()


print(f"\nBest Model: {best_model_name} with cross-validated accuracy {best_accuracy:.2f}")


# Neural Network Implementation
def create_nn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification output layer
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the neural network
nn_model = create_nn(X_train.shape[1])
nn_history = nn_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate the neural network
nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\nNeural Network Accuracy: {nn_accuracy:.2f}")

# Advanced Metrics and Curves
y_prob = nn_model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'Neural Network (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Compare models
print("\nComparison of Model Performances:")
print(f"{best_model_name} Cross-Validated Accuracy: {best_accuracy:.2f}")
print(f"Neural Network Accuracy: {nn_accuracy:.2f}")


# Dictionary to store model performance
model_performance = {}

# Loop through models and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    # Store metrics
    model_performance[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.2f}")

# Display the results for all models in a DataFrame
performance_df = pd.DataFrame(model_performance).T
print("\nModel Comparison:\n", performance_df)

# Identify the best model based on F1 score (can be customized)
best_model_name = performance_df['F1 Score'].idxmax()
print(f"\nBest Model: {best_model_name}")

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC: {roc_auc_score(y_test, y_pred_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

