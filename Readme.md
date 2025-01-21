# Titanic Survival Prediction Project

## 1. Exploratory Data Analysis (EDA)

### Dataset Overview
The Titanic dataset includes demographic and passenger details to predict survival rates. The dataset contains attributes such as passenger class, gender, age, fare, and number of family members aboard. Initial inspection of the dataset reveals:

- **Missing Values:**
  - `Age`: Contains missing values, which could impact model accuracy.
  - `Embarked`: Missing values in a small proportion.
  - `Cabin`: High proportion of missing values; decision made to drop.

### Summary Statistics
Key observations from the dataset include:

- **Survival Rate:**
  - Overall survival rate is around 38%.
  - Higher survival rates observed among females and first-class passengers.

- **Passenger Class Influence:**
  - First-class passengers had a significantly higher survival rate than third-class passengers.

- **Gender Impact:**
  - Females had a much higher survival rate than males.

- **Age Distribution:**
  - Younger passengers had a higher survival rate compared to older ones.

### Visualizations
Exploratory visualizations provided insights such as:

- Bar plots showing survival distribution across gender, class, and embarkation ports.
- Box plots highlighting fare and age distributions across survival outcomes.
- Correlation heatmap illustrating relationships between variables.

---

## 2. Data Preparation for ML

### Handling Missing Values

- `Age`: Filled missing values using median imputation.
- `Embarked`: Filled missing values using mode imputation.

### Feature Engineering

- Created `FamilySize` by summing `SibSp` and `Parch` to capture family influence.
- Added `IsAlone` feature to indicate passengers traveling alone.
- Derived `Title`, `Fare_Per_Person`, `AgeBin`, and `FareBin`.
- Used one-hot encoding for additional categorical features.

### Encoding Categorical Variables

- Converted `Sex` and `Embarked` to numerical values using one-hot encoding.

### Feature Selection

Dropped unnecessary columns: `PassengerId`, `Name`, `Ticket`, `Cabin`, which were not predictive.

### Data Splitting

The dataset was split into training and test sets using an 80-20 split.

### Data Imbalance:
- Applied SMOTE to address class imbalance in the Survived column.

### Advanced Metrics:
- Calculated ROC-AUC for the neural network.
- Plotted the ROC Curve for visual evaluation.

---

## 3. ML Model Development

### Models Evaluated

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Support Vector Machine (SVM)**
4. **Random Forest Classifier**
5. **Gradient Boosting Classifier**

### Model Evaluation

Performance metrics were assessed using:

- **Accuracy Score:** Measures overall correct predictions.
- **Classification Report:** Provides precision, recall, and F1-score.
- **Confusion Matrix:** Evaluates true vs. false positives/negatives.

### Model Optimization:
- Used cross-validation to evaluate models more reliably.
- Prepared the framework for hyperparameter tuning (e.g., GridSearchCV).

### Key Observations of Feature Correlation Heatmap

1. Survival Correlation
 - Survived vs Sex_male: Strong negative correlation (-0.54). This suggests that being male is negatively associated with survival, indicating females were more likely to survive (consistent with "women and children first" during evacuation).
 - Survived vs Pclass: Moderate negative correlation (-0.34). Passengers in higher classes (e.g., 1st class) were more likely to survive.
 - Survived vs Fare: Moderate positive correlation (0.26). Passengers who paid higher fares (likely in better classes) had higher survival rates.
 - Survived vs Title_Mrs and Title_Miss: Positive correlation, suggesting that women (especially younger or married ones) had better survival chances.
 - Survived vs IsAlone: Weak negative correlation. Passengers traveling alone might have had lower survival rates.
2. Feature Relationships
 - Pclass vs Fare: Strong negative correlation (-0.55). Higher ticket classes correlate with higher fares.
 - FamilySize vs SibSp and Parch: Strong positive correlations (0.89 and ~0.78, respectively). Family size is directly influenced by the number of siblings/spouses and parents/children aboard.
 - FareBin_VeryHigh vs Pclass: Strong negative correlation (-0.66). This highlights the relationship between high fares and first-class passengers.
3. High Correlations (Potential Multicollinearity)
 - SibSp, Parch, and FamilySize: Features are closely related. Including all might introduce redundancy in models sensitive to multicollinearity.
 - FareBin categories and Fare: Strong correlations exist between these derived features, as expected.
 - AgeBin categories and Age: Strong correlation among binned age categories and raw age values.


### Best Model Selection

After evaluation, the best-performing model was:

- **Gradient Boosting Classifier** with an accuracy of 83%, demonstrating superior predictive power.

The selected model was applied to the test dataset and final predictions were saved.

### Neural Network:

- Included a fully connected neural network and evaluated its performance alongside traditional models.

### Comparison:

1. Gradient Boosting Model:
 - Performance Metrics:
    - Accuracy: ~0.90
    - AUC (Area Under the Curve): 0.95
    - F1 Score: ~0.90
 - Observation: This model achieved the highest AUC and accuracy, suggesting it performs best in distinguishing between classes (Survived vs. Not Survived).
2. Other Models:
  - Logistic Regression:
    - AUC: ~0.93
    - Slightly lower accuracy compared to Gradient Boosting but still strong performance.
 - Decision Tree:
    - AUC: ~0.93
    - Strong recall but potentially prone to overfitting compared to ensemble methods.
 - Support Vector Machine (SVM):
    - AUC: ~0.91
    - Lower precision and F1 score than the top-performing models.
 - Random Forest:
    - AUC: ~0.92
    - Robust performance due to ensemble learning but slightly underperformed compared to Gradient Boosting.
3. Neural Network:
 - Performance Metrics:
    - Accuracy: 0.83
    - AUC: ~0.91
 - Observation:
    - The neural network slightly underperformed compared to Gradient Boosting and other traditional models.
    - While neural networks are powerful, they might require more hyperparameter tuning or additional data to achieve better results in this scenario.
4. ROC Curves:
  - The ROC curve for Gradient Boosting is the closest to the top-left corner, indicating its superior ability to balance sensitivity and specificity.
  - Neural Network and Logistic Regression have slightly less steep curves, showing comparable but lower discriminative power.

---

## 4. Ethical Analysis

### Introduction
The Titanic dataset provides an opportunity to analyze survival patterns during the tragic 1912 maritime disaster. However, as with any historical dataset, it reflects societal biases of its time, such as gender, class, and age disparities. Using such data for predictive modeling necessitates a thorough ethical examination.

### Identification of Biases
1. **Gender Bias:**
   - The survival rate for women (approximately 74%) was significantly higher than that for men (approximately 19%). This disparity reflects the "women and children first" policy during evacuation.

2. **Class Bias:**
   - First-class passengers had a survival rate of 62%, compared to 47% for second-class and 24% for third-class passengers. This inequality arose from the proximity of first-class cabins to lifeboats and better access to resources.

3. **Age Bias:**
   - Children had higher survival rates, highlighting prioritization of younger passengers.

### Impact of Bias on Model Fairness
Biases in the dataset can propagate into the ML model, leading to unfair predictions. For instance, a model trained on this data might systematically predict lower survival chances for third-class male passengers, regardless of other factors.

### Trade-offs Between Accuracy and Fairness
Optimizing for accuracy often involves leveraging patterns that reflect historical biases. Ensuring fairness requires addressing these biases, such as balancing the dataset or weighting predictions, potentially at the cost of reduced accuracy.

### Real-world Implications
Using biased models for decision-making could perpetuate inequalities. For example, in resource allocation scenarios, predictions from such models might disadvantage certain groups.

### Ethical Recommendations
1. **Transparency:** Clearly communicate the limitations and biases of the model.
2. **Bias Mitigation:** Explore techniques like re-sampling or fairness-aware algorithms.
3. **Contextual Awareness:** Recognize the historical context of the dataset and avoid applying predictions to modern scenarios without adjustments.
   
---

## 5. Executive Summary

### Project Objectives
The project aimed to build a predictive model to classify Titanic passengers as survivors or non-survivors based on available demographic and ticket-related information.

### Key Steps Undertaken

1. **Exploratory Data Analysis:**
   - Identified key patterns and correlations.
   
2. **Data Preparation:**
   - Addressed missing values, engineered new features, and encoded categorical data.

3. **Model Development:**
   - Compared various machine learning algorithms and selected the best one based on performance metrics.

4. **Ethical Considerations:**
   - Analyzed bias and fairness aspects related to gender and class disparities.
  
Screen Shots

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjsAAAHHCAYAAABZbpmkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAsxUlEQVR4nO3df1hUZcL/8c/wa0BhhlABKcC0NiRNNzSdJ3PVUDJsLTG1ZY3KdXcNbYvNXJ41NcssbbPNKHdbDduiyE2tNE2j1B7FH0tppmlm+mCPDlgJg7ryQ+b7R1+mJrQUgcHb9+u65rqa+9xzzn3sQt/XmTOMxe12uwUAAGAoP18vAAAAoCkROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAWow77rhDHTp0aNJjWCwWTZs2rUmPAaBlIXaAC9T27ds1fPhwxcfHKzg4WBdffLEGDhyouXPn+nppLYbL5dJDDz2kbt26KTQ0VCEhIerSpYsmTZqkgwcP+np5kqS3336beAN+goXvxgIuPBs2bFD//v0VFxenjIwMRUdH68CBA9q4caP27t2rzz//3Cfrqq6uVm1traxWa5Mdw2KxaOrUqT8ZCF988YWSk5NVXFysW2+9VX369FFQUJA+/vhjvfLKK4qIiNBnn33WZOs8U+PHj1dOTo74qxw4vQBfLwBA85sxY4bsdru2bNmi8PBwr22lpaWNdpxjx46pdevWZzw/MDCw0Y59LmpqajRs2DCVlJRozZo16tOnj9f2GTNm6PHHH/fR6gCcLd7GAi5Ae/fu1ZVXXlkvdCQpMjLS89/79++XxWJRbm5uvXk/vPdl2rRpslgs2rlzp371q1/poosuUp8+ffTEE0/IYrHof//3f+vtIzs7W0FBQTpy5Igk73t2qqurFRERoTvvvLPe61wul4KDg3X//fdLkqqqqjRlyhQlJSXJbrerdevWuu666/T++++fxZ/Kd15//XVt27ZNf/7zn+uFjiTZbDbNmDHDa2zRokVKSkpSSEiI2rZtq1//+tf6v//7P685/fr1U79+/ert74f3KtX9uT/xxBP6+9//rk6dOslqtapnz57asmWL1+tycnIkffv/o+4BwBuxA1yA4uPjVVRUpE8++aTR933rrbfq+PHjevTRRzV27FiNGDFCFotFr732Wr25r732mgYNGqSLLrqo3rbAwEDdcsstWrp0qaqqqry2LV26VJWVlRo1apSkb+PnH//4h/r166fHH39c06ZN0+HDh5WSkqKtW7ee9Tm8+eabkqTRo0ef0fzc3FyNGDFC/v7+mjlzpsaOHavFixerT58+KisrO+vj18nLy9Ps2bP1u9/9To888oj279+vYcOGqbq6WpL0u9/9TgMHDpQk/fOf//Q8AHjjbSzgAnT//fdr8ODB6t69u6655hpdd911uv7669W/f/9zfiupW7duysvL8xrr3bu38vPzNXHiRM/Yli1b9MUXX/zovTMjR47UggULtGrVKg0ZMsQznp+fr44dO6pHjx6SpIsuukj79+9XUFCQZ87YsWOVkJCguXPnav78+Wd1Dp9++qnsdrtiY2N/cm51dbUmTZqkLl26aN26dQoODpYk9enTR0OGDNGcOXP00EMPndXx6xQXF2vPnj2eGLziiis0dOhQvfPOOxoyZIgcDod+9rOfafXq1fr1r3/doGMAFwKu7AAXoIEDB6qwsFC//OUvtW3bNs2aNUspKSm6+OKLPVc1Gur3v/99vbGRI0eqqKhIe/fu9Yzl5+fLarVq6NChp93XgAED1LZtW+Xn53vGjhw5otWrV2vkyJGeMX9/f0/o1NbW6ptvvlFNTY169OihDz/88KzPweVyKSws7Izm/vvf/1ZpaanuvvtuT+hIUmpqqhISErR8+fKzPn6dkSNHel31uu666yR9e/M0gDNH7AAXqJ49e2rx4sU6cuSINm/erOzsbFVUVGj48OHauXNng/d76aWX1hu79dZb5efn54kWt9utRYsWafDgwbLZbKfdV0BAgNLS0vTGG2+osrJSkrR48WJVV1d7xY4kLVy4UFdddZWCg4PVpk0btWvXTsuXL1d5eflZn4PNZlNFRcUZza27F+mKK66oty0hIeGU9yqdqbi4OK/ndeFTd48TgDND7AAXuKCgIPXs2VOPPvqonnvuOVVXV2vRokWSdNqbXU+ePHna/YWEhNQbi4mJ0XXXXee5b2fjxo0qLi6uFyynMmrUKFVUVGjFihWSvr3PJyEhQd26dfPMeemll3THHXeoU6dOmj9/vlauXKnVq1drwIABqq2t/clj/FBCQoLKy8t14MCBs37tjznbP09/f/9TjvMxc+DsEDsAPOrugTl06JCk764k/PAm24ZcrRg5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6emnfH16erqWLl3quYrjcrm0fPlyz/z8/HzV1tbqH//4h7p27arOnTvrhRdeUHFxsecq01NPPaXs7GwNGzZMnTt31rx587gnCGiBiB0AzSotLU0HDx7Um2++qRtuuEFr1qzR1Vdfrdzc3LPaT48ePbyejxo1Svv379fGjRslfXuV5uqrr1ZCQsIpX3/jjTcqMDBQb775piTp9ddfl81mU3JysiRp27Zt+vzzzxUWFua5IhUREaETJ05o7969Ki8v16FDh9SrVy/PPgMCAuqtC4DvETsAml1wcLAGDhyoBx98UBs2bNAdd9yhqVOnys/v27+Svn/fS3V19Sn30bp1a6/n0dHRGjBggOetqby8vNNe1ZG+vWF5+PDhXvNHjhzpudH56NGjSkpK0tatW70en3322SnfGgPQchE7AHwuMTFRx44dU7t27SRJhw4d8mz7/s3KPyU9PV35+fkqLCzUF198oVGjRv3k/JUrV2rHjh167733vOLo6quv1p49exQZGanLLrvM62G322W329W+fXtt2rTJ85qamhoVFRWd8XoBNA9iB0Cz+frrrzVgwAC99NJL+vjjj7Vv3z4tWrRIs2bN0tChQxUSEqLevXt7bjxeu3atJk+efMb7HzZsmCoqKjRu3Dj1799fMTExPzq/b9++io6OVnp6ui699FKvt6TS09PVtm1bDR06VB988IH27dunNWvW6J577tGXX34pSfrDH/6gxx57TEuXLtWuXbt09913N/kvLQRw9ogdAM0mNDRUvXr10pw5c9S3b1916dJFDz74oMaOHatnnnlGkrRgwQLV1NQoKSlJ9957rx555JEz3n9YWJhuuukmbdu27UffwqpjsVh02223nXJ+q1attG7dOsXFxXluQB4zZoxOnDghm80mSfrjH/+o0aNHKyMjQw6HQ2FhYbrlllvO4k8EQHOwuPmlEAAAwGBc2QEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABjt/wE2pT7t3UcCywAAAABJRU5ErkJggg==" alt="Image Description">

### Key Results

- The Gradient Boosting Classifier achieved the highest accuracy of 83%, making it the best-suited model for deployment.
- Key survival indicators include passenger class, gender, and family size.
