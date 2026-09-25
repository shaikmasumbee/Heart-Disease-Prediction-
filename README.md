# ❤️ Heart Disease Prediction — End-to-End Machine Learning Pipeline

An end-to-end **Heart Disease Prediction** project that uses machine learning classification algorithms to predict whether a patient is likely to have heart disease based on clinical features.

The project is built as a complete ML pipeline covering **data preprocessing, transformation, feature selection, class balancing, feature scaling, model training, model evaluation, model serialization, and Flask deployment**.

---

## 📌 Project Overview

Heart disease is one of the major health conditions that requires early identification and risk assessment.

This project uses patient medical attributes such as **age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and other clinical measurements** to build classification models.

The project does not rely on a single algorithm. Multiple classification models are trained and evaluated using the same processed dataset, and their performance is compared using several evaluation metrics.

The final trained model is saved and integrated into a **Flask web application**, allowing users to enter patient information and receive a prediction.

> **Note:** This project is intended for educational and machine-learning demonstration purposes. It is not a medical diagnostic system.

---

# 🎯 Objectives

The main objectives of this project are:

* Load and inspect the heart disease dataset.
* Separate independent and dependent variables.
* Split the data into training and testing sets.
* Apply **Yeo-Johnson transformation** to numerical features.
* Perform feature selection using **SelectKBest with ANOVA F-test**.
* Handle class imbalance using **SMOTE**.
* Standardize features using **StandardScaler**.
* Train multiple classification algorithms.
* Evaluate models using multiple performance metrics.
* Generate and save a combined **ROC-AUC curve**.
* Save the trained model and scaler using Pickle.
* Build a Flask-based prediction application.
* Deploy the trained model as a web application.

---

# 📊 Dataset

The project uses a heart disease dataset containing **303 records and 14 columns**.

The dataset contains **13 input features** and **1 target variable**.

### Input Features

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex of the patient                    |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting electrocardiographic results  |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia-related measurement       |

### Target

The final column is used as the dependent variable:

* `0` → No heart disease
* `1` → Heart disease

All features used in this implementation are numerical, so categorical encoding is not required.

---

# 🏗️ Machine Learning Pipeline

The project follows this processing flow:

```text
                    heart.csv
                        │
                        ▼
                Load Dataset
                        │
                        ▼
              Data Inspection
                        │
                        ▼
              Train/Test Split
                 80% / 20%
                        │
                        ▼
           Numerical Data Separation
                        │
                        ▼
          Yeo-Johnson Transformation
                        │
                        ▼
              Feature Selection
                 SelectKBest
                        │
                        ▼
              SMOTE Balancing
                        │
                        ▼
              Standard Scaling
                        │
                        ▼
              Train ML Models
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Logistic      Decision      Random
      Regression       Tree         Forest
          │             │             │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
         KNN            SVM       Gradient
                                   Boosting
                        │
                        ▼
                Model Evaluation
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Accuracy   F1 Score   ROC-AUC
                        │
                        ▼
                  ROC Curve
                        │
                        ▼
             Final Model Selection
                        │
                        ▼
             Logistic Regression
                        │
                        ▼
              Heart_Disease_Model.pkl
                        │
                        ▼
                 Flask Application
                        │
                        ▼
                  Prediction
```

---

# 🔬 Data Preprocessing

## 1. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A fixed `random_state=42` is used for reproducibility.

Stratified splitting is also applied so that the target-class distribution is maintained between training and testing data.

```python
train_test_split(
    self.X,
    self.y,
    test_size=0.20,
    random_state=42,
    stratify=self.y
)
```

---

## 2. Yeo-Johnson Transformation

The project applies **Yeo-Johnson Power Transformation** to the training features.

The transformer is fitted only on the training data and then used to transform both training and testing data.

```python
PowerTransformer(
    method="yeo-johnson"
)
```

### Why?

The transformation can make numerical feature distributions more suitable for machine-learning algorithms by reducing skewness and stabilizing variance.

The transformation is implemented in:

```text
yeo_timing.py
```

---

# 🧮 Feature Selection

Feature selection is performed using:

```python
SelectKBest(
    score_func=f_classif,
    k="all"
)
```

The project uses the **ANOVA F-test (`f_classif`)** to evaluate the relationship between each feature and the target.

Currently, `k="all"` means all features are retained after being evaluated.

The feature-selection logic is implemented in:

```text
fs.py
```

---

# ⚖️ Handling Class Imbalance

The project uses **SMOTE — Synthetic Minority Over-sampling Technique**.

```python
SMOTE(
    random_state=42
)
```

SMOTE generates synthetic samples for the minority class instead of simply duplicating existing records.

This is applied **only to the training data**.

```text
Training Data
      │
      ▼
    SMOTE
      │
      ▼
Balanced Training Data
```

The testing dataset is not oversampled because the test set should represent unseen data.

---

# 📏 Feature Scaling

After balancing the training data, **StandardScaler** is used.

```python
StandardScaler()
```

The scaler is fitted on the balanced training data:

```python
sc.fit(X_train_bal)
```

The same scaler is then used to transform both training and testing data.

The fitted scaler is saved as:

```text
scaled_model.pkl
```

This allows the Flask application to apply the **same preprocessing scale** to new user inputs before making predictions.

---

# 🤖 Machine Learning Models

Six classification algorithms are trained and evaluated.

### 1. Logistic Regression

Used as a linear classification model and selected as the final model in the current implementation.

### 2. Decision Tree

A tree-based model that makes predictions using a sequence of feature-based decisions.

### 3. Random Forest

An ensemble of multiple decision trees designed to improve generalization compared with an individual tree.

### 4. K-Nearest Neighbors — KNN

Classifies a sample based on the classes of nearby training samples.

### 5. Support Vector Machine — SVM

Finds a decision boundary that separates classes in feature space.

### 6. Gradient Boosting

Builds an ensemble of weak learners sequentially, with later models focusing on correcting previous errors.

---

# 📈 Model Evaluation

Each trained model is evaluated using multiple metrics.

## Accuracy

Measures the overall percentage of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

## Precision

Measures how many predicted positive cases were actually positive.

Important when false-positive predictions matter.

## Recall

Measures how many actual positive cases were correctly identified.

For a heart-disease classification problem, recall is particularly useful because missing an actual positive case can be important.

## F1 Score

The harmonic mean of precision and recall.

It provides a balance between the two metrics.

## Confusion Matrix

The project generates a confusion matrix for every model.

It contains:

```text
                 Predicted
                 0       1
Actual  0       TN      FP
        1       FN      TP
```

## ROC-AUC

ROC-AUC is also calculated for every model.

The project uses:

```python
roc_auc_score()
```

and

```python
roc_curve()
```

to calculate and generate ROC information.

---

# 📊 ROC Curve

A combined ROC curve is generated for all six models.

The resulting visualization is saved as:

```text
roc_curve.png
```

The graph displays:

* False Positive Rate on the X-axis
* True Positive Rate on the Y-axis
* ROC-AUC value for each model
* Random classifier reference line

Example output:

```text
ROC Curve - Heart Disease Classification
```

This makes it easier to visually compare the classification performance of the trained models.

---

# 🏆 Final Model

After comparing the trained models, the current implementation uses **Logistic Regression as the final model**.

The final model is retrained using the processed training data and saved as:

```text
Heart_Disease_Model.pkl
```

The saved model is later loaded by the Flask application.

### Important

The final-model choice is currently hard-coded as Logistic Regression:

```python
final_model = LogisticRegression(
    max_iter=1000
)
```

So the README should not claim that the application automatically selects whichever model has the highest ROC-AUC. In the current code, **the comparison is performed, but Logistic Regression is explicitly chosen afterward**.

---

# 🌐 Flask Web Application

The trained model is integrated into a Flask application.

The application:

1. Displays a prediction form.
2. Accepts patient information.
3. Converts the input values into numerical values.
4. Creates a NumPy array.
5. Applies the saved `StandardScaler`.
6. Sends the scaled data to the trained model.
7. Generates a prediction.
8. Displays the result on the webpage.

### Prediction Output

The application displays either:

```text
No Heart Disease
```

or

```text
Heart Disease Detected
```

---

# 🗂️ Project Structure

```text
MINI_PROJECT_2/
│
├── app.py
├── main.py
├── all_models.py
├── fs.py
├── yeo_timing.py
├── log_code.py
│
├── heart.csv
│
├── Heart_Disease_Model.pkl
├── scaled_model.pkl
│
├── roc_curve.png
│
├── requirements.txt
├── Procfile
│
├── logs/
│   ├── main.log
│   ├── all_models.log
│   ├── fs.log
│   └── yeo_timing.log
│
└── templates/
    └── index.html
```

---

# 📄 File Description

| File                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `main.py`                 | Controls the complete ML preprocessing and training pipeline |
| `all_models.py`           | Trains and evaluates all classification models               |
| `yeo_timing.py`           | Performs Yeo-Johnson transformation                          |
| `fs.py`                   | Performs feature selection                                   |
| `log_code.py`             | Configures project logging                                   |
| `app.py`                  | Flask web application for prediction                         |
| `heart.csv`               | Heart disease dataset                                        |
| `Heart_Disease_Model.pkl` | Serialized final Logistic Regression model                   |
| `scaled_model.pkl`        | Serialized StandardScaler                                    |
| `roc_curve.png`           | ROC curve comparison of trained models                       |
| `requirements.txt`        | Python dependencies                                          |
| `Procfile`                | Deployment configuration for Gunicorn                        |
| `logs/`                   | Stores execution and error logs                              |
| `templates/index.html`    | Front-end prediction interface                               |

---

# 📝 Logging

Logging has been implemented throughout the project to monitor the ML pipeline.

Separate log files are maintained for different modules.

### `main.log`

Contains information related to:

* Dataset loading
* Dataset dimensions
* Null-value checks
* Train-test split
* Transformation
* Feature selection
* SMOTE
* Scaling
* Model training

### `all_models.log`

Contains:

* Model names
* Training status
* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC
* Confusion matrix
* ROC curve generation
* Final model saving

### `fs.log`

Contains feature-selection information.

### `yeo_timing.log`

Contains Yeo-Johnson transformation information and errors.

This logging structure makes debugging easier than relying only on console output.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Imbalanced-learn

### Visualization

* Matplotlib

### Web Framework

* Flask

### Model Serialization

* Pickle

### Deployment

* Gunicorn

### Development Tools

* PyCharm / VS Code
* Git
* GitHub

---

# 📦 Python Libraries

The major libraries used in the project include:

```text
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
flask
gunicorn
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd MINI_PROJECT_2
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the ML Pipeline

Run:

```bash
python main.py
```

The pipeline will:

```text
Load Dataset
      ↓
Train/Test Split
      ↓
Yeo-Johnson Transformation
      ↓
Feature Selection
      ↓
SMOTE
      ↓
Standard Scaling
      ↓
Train Six Models
      ↓
Evaluate Models
      ↓
Generate ROC Curve
      ↓
Save Scaler
      ↓
Save Final Model
```

The following files will be generated/updated:

```text
Heart_Disease_Model.pkl
scaled_model.pkl
roc_curve.png
logs/*.log
```

---

## 5. Run the Flask Application

After the model and scaler have been generated:

```bash
python app.py
```

The Flask application will start locally.

Open the local Flask address shown in the terminal and enter the patient details into the prediction form.

---

# 🚀 Deployment

The project includes a `Procfile` configured for Gunicorn:

```text
web: gunicorn app:app
```

This allows the Flask application to be started using Gunicorn on supported hosting platforms.

The deployment requires:

```text
app.py
requirements.txt
Procfile
Heart_Disease_Model.pkl
scaled_model.pkl
templates/
```

---

# 🔐 Important Implementation Details

### No data leakage during transformation

The Yeo-Johnson transformer is fitted on the training data and then applied to the test data.

### No SMOTE on test data

SMOTE is applied only to the training data.

### Scaler fitted on training data

The `StandardScaler` is fitted using the balanced training dataset and then reused for test/new data.

### Same scaler during prediction

The Flask application loads the saved scaler instead of creating a new scaler for every prediction.

These steps help keep the training and prediction pipelines consistent.

---

# 💡 Key Learning Outcomes

Through this project, the following machine-learning concepts are demonstrated:

* End-to-end ML pipeline development
* Train-test splitting
* Stratified sampling
* Numerical preprocessing
* Yeo-Johnson transformation
* Feature selection
* Class imbalance handling
* SMOTE
* Standardization
* Multiple classification algorithms
* Model evaluation
* Confusion matrix
* ROC curve
* ROC-AUC
* Model serialization
* Flask model integration
* Logging
* Basic ML deployment

---

# 🔄 Complete Project Workflow

```text
                    ┌─────────────────┐
                    │    heart.csv    │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Data Inspection   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Train/Test Split  │
                  │       80 / 20       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Yeo-Johnson        │
                  │  Transformation     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Feature Selection  │
                  │     SelectKBest      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │       SMOTE         │
                  │ Training Balance    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  StandardScaler     │
                  └──────────┬──────────┘
                             │
                             ▼
             ┌───────────────┴────────────────┐
             │                                │
             ▼                                ▼
    ┌──────────────────┐            ┌─────────────────┐
    │  Six ML Models   │            │  ROC-AUC /      │
    │                  │            │  Other Metrics  │
    └────────┬─────────┘            └────────┬────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
                    ┌──────────────────┐
                    │   ROC Curve      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Logistic          │
                    │ Regression        │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Heart_Disease_      │
                  │ Model.pkl           │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Flask App        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Patient Prediction  │
                  └─────────────────────┘
```

---

# ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**.

The predictions generated by this application should **not be considered medical advice, diagnosis, or a substitute for professional medical evaluation**.

---

# 👩‍💻 Author

**Shaik Masumbee**

---

# 👨‍🏫 Guidance

**Kamal Sir**
**ViharaTech Institute**

---

# 🚀 Live Demo

**Render:** [View Live Application](https://heart-disease-prediction-1-eddv.onrender.com/)

## ⭐ Project Highlights

```text
✔ End-to-End Machine Learning Pipeline
✔ Multiple Classification Algorithms
✔ Yeo-Johnson Transformation
✔ Feature Selection
✔ SMOTE Class Balancing
✔ StandardScaler
✔ Accuracy / Precision / Recall / F1
✔ Confusion Matrix
✔ ROC-AUC Comparison
✔ ROC Curve Visualization
✔ Model Serialization
✔ Flask Web Application
✔ Logging & Error Handling
✔ Deployment Configuration
```

If you found this project useful, consider giving the repository a ⭐ on GitHub.
