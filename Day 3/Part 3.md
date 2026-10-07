# DAY 3 — PART 3: Classification Train / Test Split & Accuracy

> **Overview:** In Day 2, we evaluated regression models using continuous error metrics. In Day 3, we learn how to evaluate classification models using `train_test_split` and our first classification evaluation metric: **Accuracy**. We will cover the intuition behind train/test split, manual accuracy calculation, Scikit-learn code, and the crucial pitfall known as the "Accuracy Paradox"! 🎯

---

## 📋 Table of Contents

1. [👶 Intuition: Why Split Data into Train and Test Sets?](#1️⃣-intuition-why-split-data-into-train-and-test-sets)
2. [⚙️ How `train_test_split()` Works in Scikit-Learn](#2️⃣-how-train_test_split-works-in-scikit-learn)
3. [📊 End-to-End Classification Workflow](#3️⃣-end-to-end-classification-workflow)
4. [🎯 What is Accuracy? (Mathematical Concept)](#4️⃣-what-is-accuracy-mathematical-concept)
5. [🐍 Implementing `accuracy_score` in Python](#5️⃣-implementing-accuracy_score-in-python)
6. [💻 Complete End-to-End Code](#6️⃣-complete-end-to-end-code)
7. [⚖️ Regression vs Classification Evaluation](#7️⃣-regression-vs-classification-evaluation)
8. [⚠️ The Accuracy Paradox (When Accuracy Lies!)](#8️⃣-the-accuracy-paradox-when-accuracy-lies)
9. [🧠 Core Memory Points](#-core-memory-points)
10. [🎯 Top Interview Questions & Answers](#-top-interview-questions--answers)

---

## 1️⃣ Intuition: Why Split Data into Train and Test Sets?

Suppose we have the following dataset:

| Study Hours ($X$) | Actual Result ($y$) |
| :---: | :---: |
| 1 | Fail (0) |
| 2 | Fail (0) |
| 3 | Fail (0) |
| 4 | Pass (1) |
| 5 | Pass (1) |
| 6 | Pass (1) |
| 7 | Pass (1) |
| 8 | Pass (1) |

If we train our model on **all available data**, the model might simply memorize the answers (Overfitting). We would have no way to verify whether the model works accurately on **new, unseen data**!

```text
                     Original Dataset (8 Students)
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
          Training Set (75%)                  Testing Set (25%)
             (6 Students)                        (2 Students)
                 │                                   │
                 ▼                                   ▼
         Model Learns Here                   Evaluated on Unseen Data
            (model.fit)                       (model.predict)
```

> 💡 **Analogy:**  
> * **Training Data:** Practice questions solved before an exam.  
> * **Testing Data:** The final exam questions that test real understanding!

---

## 2️⃣ How `train_test_split()` Works in Scikit-Learn

We import `train_test_split` from the `sklearn.model_selection` module:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.25, 
    random_state=42
)
```

### Parameter Breakdown:
* **`x` & `y`:** The original feature matrix and target labels.
* **`test_size=0.25`:** 25% of the data is reserved for testing, and the remaining 75% is used for training.
* **`random_state=42`:** Sets a random seed so that the data split is identical every time you run the script.

### Resulting Variables:
| Variable | Description | Purpose |
| :--- | :--- | :--- |
| `x_train` | Training features (Study hours) | Features provided for model training |
| `y_train` | Training labels (Pass / Fail) | Correct labels used to train the model |
| `x_test` | Testing features (Unseen study hours) | Inputs used to evaluate the model |
| `y_test` | Testing ground truth labels | Actual labels used to compare predictions |

---

## 3️⃣ End-to-End Classification Workflow

```text
       Raw Data (x, y)
              │
              ▼
    [ train_test_split() ]
              │
      ┌───────┴───────┐
      ▼               ▼
(x_train, y_train)  (x_test, y_test)
      │               │
      ▼               │
 [ model.fit() ]      │
      │               │
      ▼               ▼
[ Trained Model ] ──> [ model.predict(x_test) ]
                              │
                              ▼
                         Predictions
                              │
                              ▼
                 [ accuracy_score(y_test, predictions) ]
```

---

## 4️⃣ What is Accuracy? (Mathematical Concept)

Suppose our test set contains 4 students:

* **Actual Labels ($y_{test}$):** `[0, 1, 1, 0]`
* **Model Predictions ($\hat{y}$):** `[0, 1, 0, 0]`

### Comparison Table:
| Student # | Actual Label ($y_{test}$) | Model Prediction ($\hat{y}$) | Result |
| :---: | :---: | :---: | :---: |
| Student 1 | `0` (Fail) | `0` (Fail) | ✅ Correct |
| Student 2 | `1` (Pass) | `1` (Pass) | ✅ Correct |
| Student 3 | `1` (Pass) | `0` (Fail) | ❌ Wrong |
| Student 4 | `0` (Fail) | `0` (Fail) | ✅ Correct |

### Formula:
$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$$

$$\text{Accuracy} = \frac{3}{4} = 0.75 \implies 75\%$$

---

## 5️⃣ Implementing `accuracy_score` in Python

We import `accuracy_score` from `sklearn.metrics`:

```python
from sklearn.metrics import accuracy_score

# Compute accuracy
accuracy = accuracy_score(y_test, prediction)
print("Accuracy (Decimal)   :", accuracy)
print("Accuracy (Percentage):", accuracy * 100, "%")
```

### Output:
```text
Accuracy (Decimal)   : 0.75
Accuracy (Percentage): 75.0 %
```

---

## 6️⃣ Complete End-to-End Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Dataset
x = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1, 1, 1]

# 2. Train/Test Split (75% Train, 25% Test)
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.25, 
    random_state=42
)

# 3. Model Creation
model = LogisticRegression()

# 4. Training on Training Data Only
model.fit(x_train, y_train)

# 5. Making Predictions on Test Data
prediction = model.predict(x_test)

# 6. Evaluation
accuracy = accuracy_score(y_test, prediction)

print("Actual Test Labels    :", y_test)
print("Model Predictions     :", prediction)
print(f"Model Accuracy        : {accuracy * 100:.2f}%")
```

---

## 7️⃣ Regression vs Classification Evaluation

| Aspect | Regression (Day 2) | Classification (Day 3) |
| :--- | :--- | :--- |
| **Problem Type** | Continuous numerical prediction | Category / Class prediction |
| **Output Type** | Real numbers ($34.5$, $82.0$) | Discrete labels ($0$ or $1$, Spam/Ham) |
| **Core Metrics** | **MSE** (Mean Squared Error)<br>**MAE** (Mean Absolute Error)<br>**$R^2$ Score** | **Accuracy Score**<br>**Precision**<br>**Recall**<br>**F1 Score** |
| **Measurement Goal** | Minimize distance error ($y - \hat{y}$) | Maximize correct class count |

---

## 8️⃣ The Accuracy Paradox (When Accuracy Lies!)

> 🚨 **Critical Machine Learning Concept:**  
> A high accuracy score does not automatically mean a model is reliable!

### The Imbalanced Dataset Trap:
Consider a rare disease screening dataset with **100 patients**:
* **95 patients** $\longrightarrow$ Healthy (0)
* **5 patients** $\longrightarrow$ Disease Positive (1)

Suppose we have a naive model that predicts **every single patient is Healthy (0)**:

```text
Actual Data:        95 Healthy,  5 Disease
Dummy Prediction:  100 Healthy,  0 Disease

Correct Predictions = 95 / 100 = 95% Accuracy!
```

### The Problem:
* The accuracy is **95%**, but the model failed to detect **a single patient with the disease (0% caught)!**
* In healthcare and safety-critical domains, this mistake is unacceptable.
* This is why we need more descriptive evaluation metrics:
  * **Confusion Matrix**
  * **Precision**
  * **Recall**
  * **F1 Score**

---

## 🧠 Core Memory Points

```text
Data Split (train_test_split)
            ↓
Train Data Only (model.fit)
            ↓
Test Input Feed (model.predict)
            ↓
Actual vs Predicted (accuracy_score)
            ↓
Accuracy = (Correct / Total) * 100
```

1. **Never evaluate on training data:** Doing so creates an overly optimistic and biased view of model performance.
2. **`test_size`:** Typically set between 20% (`0.20`) and 30% (`0.30`).
3. **`random_state`:** Ensures reproducible splits across different runs.
4. **Accuracy limitation:** Highly misleading when classes are imbalanced.

---

## 🎯 Top Interview Questions & Answers

### Q1. What is Accuracy in classification?
> **Answer:**  
> Accuracy is the fraction of total predictions that the model got right:
> $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

### Q2. Why do we split datasets into train and test sets?
> **Answer:**  
> To prevent data leakage and detect overfitting. Splitting allows us to measure how well the model generalizes to new, unseen data.

### Q3. What is the 'Accuracy Paradox' (Class Imbalance Problem)?
> **Answer:**  
> When classes are heavily skewed (e.g., 99% negative, 1% positive), a trivial model predicting only the majority class achieves 99% accuracy while failing entirely on the minority class. In such cases, metrics like Precision, Recall, and F1 Score are required.