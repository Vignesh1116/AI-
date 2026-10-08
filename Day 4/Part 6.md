# DAY 4 — PART 6: End-to-End Decision Tree Mini-Project

> **Overview:** In this concluding part of Day 4, we bring together every single concept we have learned—splits, Gini impurity, depth control, evaluation, and tree plotting—into a complete, end-to-end **Student Pass/Fail Prediction Project**. We will walk through data preparation, model training, performance evaluation, inference on unseen student data, and full tree visualization with professional Python code. 🎓🚀

---

## 📋 Table of Contents

1. [🎯 Project Objective & Problem Statement](#1️⃣-project-objective--problem-statement)
2. [🪜 Step-by-Step Implementation Guide](#2️⃣-step-by-step-implementation-guide)
   - [Step 1: Import Libraries](#step-1-import-libraries)
   - [Step 2: Define Feature Data and Labels](#step-2-define-feature-data-and-labels)
   - [Step 3: Split into Train and Test Sets](#step-3-split-into-train-and-test-sets)
   - [Step 4: Instantiate Decision Tree Classifier](#step-4-instantiate-decision-tree-classifier)
   - [Step 5: Train the Model](#step-5-train-the-model)
   - [Step 6: Generate Predictions on Test Set](#step-6-generate-predictions-on-test-set)
   - [Step 7: Evaluate Model Performance](#step-7-evaluate-model-performance)
   - [Step 8: Predict for a New Student (Inference)](#step-8-predict-for-a-new-student-inference)
   - [Step 9: Visualize the Trained Decision Tree](#step-9-visualize-the-trained-decision-tree)
3. [💻 Complete Self-Contained Project Script](#3️⃣-complete-self-contained-project-script)
4. [📊 Expected Output & Interpretation](#4️⃣-expected-output--interpretation)
5. [🔄 The Universal Machine Learning Workflow](#5️⃣-the-universal-machine-learning-workflow)
6. [💼 Day 4 Master Interview Questions & Answers](#-day-4-master-interview-questions--answers)
7. [🎉 Day 4 Completion & Achievements](#-day-4-completion--achievements)

---

## 1️⃣ Project Objective & Problem Statement

### 🎯 Objective:
Build an automated classification pipeline using a **Decision Tree Classifier** to predict whether a student will **Pass** or **Fail** based on their daily study hours.

### 📐 Problem Setup:
* **Feature ($X$):** Daily Study Hours (continuous numerical value)
* **Target ($y$):** Examination Result (Binary: `0 = Fail`, `1 = Pass`)
* **Algorithm:** `DecisionTreeClassifier` with `max_depth=3` to avoid overfitting

---

## 2️⃣ Step-by-Step Implementation Guide

### Step 1: Import Libraries
We import all necessary functions from `scikit-learn` for splitting, model construction, evaluation, and plotting, as well as `matplotlib` for visualization.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
```

---

### Step 2: Define Feature Data and Labels
We define $10$ student data points. Study hours are provided as a 2D list (`X`), and exam outcomes are provided as labels (`y`):

```python
# Feature X: Daily Study Hours (2D list)
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# Target y: 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

---

### Step 3: Split into Train and Test Sets
We divide the dataset into an **$80\%$ training set** and a **$20\%$ test set** using `train_test_split`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 20% reserved for testing
    random_state=42      # Reproducible shuffle split
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
```

---

### Step 4: Instantiate Decision Tree Classifier
We initialize `DecisionTreeClassifier` with `max_depth=3` to constrain depth and prevent the tree from overfitting:

```python
model = DecisionTreeClassifier(
    criterion="gini",     # Default: Gini impurity splitting
    max_depth=3,          # Limit tree depth to 3 levels
    random_state=42       # Reproducible results
)
```

---

### Step 5: Train the Model
We call `.fit()` to allow the algorithm to compute candidate splits, evaluate Gini impurity, and construct the decision rules:

```python
model.fit(X_train, y_train)
print("Model training completed successfully.")
```

---

### Step 6: Generate Predictions on Test Set
We generate predictions on the unseen test dataset:

```python
y_pred = model.predict(X_test)

print("Actual Test Labels:     ", y_test)
print("Predicted Test Labels:  ", list(y_pred))
```

---

### Step 7: Evaluate Model Performance
We compute our suite of standard classification metrics:

```python
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- MODEL EVALUATION METRICS ---")
print(f"Accuracy:         {accuracy * 100:.2f}%")
print(f"Precision:        {precision:.2f}")
print(f"Recall:           {recall:.2f}")
print(f"F1 Score:         {f1:.2f}")
print("Confusion Matrix:\n", cm)
```

---

### Step 8: Predict for a New Student (Inference)
Now suppose a new student joins and studies for **$7$ hours daily**. We run inference:

```python
new_student_hours = [[7]]
new_prediction = model.predict(new_student_hours)
result_label = "Pass" if new_prediction[0] == 1 else "Fail"

print(f"\nPrediction for new student ({new_student_hours[0][0]} hrs): {result_label} (Class {new_prediction[0]})")
```

---

### Step 9: Visualize the Trained Decision Tree
Finally, we render the learned tree structure graphically:

```python
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)
plt.title("Day 4: Student Pass/Fail Decision Tree", fontsize=14, pad=12)
plt.tight_layout()
plt.show()
```

---

## 3️⃣ Complete Self-Contained Project Script

Here is the entire, fully executable Python script ready to copy, run, or deploy:

```python
"""
Day 4 Mini-Project: Student Pass/Fail Prediction using Decision Tree Classifier
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 1. Dataset Preparation
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Model Creation
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

# 4. Model Training
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("=" * 45)
print("       STUDENT CLASSIFICATION REPORT")
print("=" * 45)
print(f"Actual Labels:     {y_test}")
print(f"Predicted Labels:  {list(y_pred)}")
print(f"Accuracy:          {accuracy * 100:.2f}%")
print(f"Precision:         {precision:.2f}")
print(f"Recall:            {recall:.2f}")
print(f"F1 Score:          {f1:.2f}")
print("Confusion Matrix:")
print(cm)
print("=" * 45)

# 6. Predict on New Unseen Sample
new_student = [[7]]
prediction = model.predict(new_student)
status = "Pass" if prediction[0] == 1 else "Fail"
print(f"\nNew Prediction (7 Study Hours): {status} [Class {prediction[0]}]")

# 7. Visualize the Tree
plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)
plt.title("Student Pass/Fail Decision Tree", fontsize=14, pad=12)
plt.tight_layout()
plt.show()
```

---

## 4️⃣ Expected Output & Interpretation

### Console Output:
```text
=============================================
       STUDENT CLASSIFICATION REPORT
=============================================
Actual Labels:     [1, 1]
Predicted Labels:  [1, 1]
Accuracy:          100.00%
Precision:         1.00
Recall:            1.00
F1 Score:          1.00
Confusion Matrix:
[[2]]
=============================================

New Prediction (7 Study Hours): Pass [Class 1]
```

### Interpretation:
1. **Accuracy (100%):** The test samples were correctly predicted.
2. **New Student Inference:** A student studying $7$ hours is classified as **Pass** because $7$ is greater than the learned split threshold ($3.5$).
3. **Visualization:** The generated plot clearly shows the root decision condition, Gini scores, sample distribution counts, and colored classifications.

---

## 5️⃣ The Universal Machine Learning Workflow

Notice how this project follows the exact end-to-end Machine Learning life cycle:

```text
 1. Problem Formulation       (Classify student outcome: Pass or Fail)
            │
            ▼
 2. Dataset Preparation       (Features: Study Hours; Labels: 0 or 1)
            │
            ▼
 3. Train / Test Split        (Prevent data leakage, validate on test set)
            │
            ▼
 4. Model Selection           (DecisionTreeClassifier with max_depth=3)
            │
            ▼
 5. Model Training (fit)      (Find best splits using Gini Impurity)
            │
            ▼
 6. Predictions (predict)     (Classify test set samples)
            │
            ▼
 7. Evaluation                (Accuracy, Confusion Matrix, Precision, Recall, F1)
            │
            ▼
 8. Interpret & Visualize     (Inspect rules with plot_tree)
            │
            ▼
 9. New Data Inference        (Make predictions on brand new students)
```

This universal workflow applies to every supervised learning task across the industry.

---

## 💼 Day 4 Master Interview Questions & Answers

### Q1: What is a Decision Tree and how does it make predictions?
> **Answer:** A Decision Tree is a supervised machine learning algorithm that segments feature space into discrete regions using hierarchical if-else decision rules learned from training data. To predict, an input sample follows branches from the root down to a leaf node based on whether it satisfies each node's condition.

### Q2: What is Gini Impurity, and how is it used during tree construction?
> **Answer:** Gini Impurity ($1 - \sum p_i^2$) measures the heterogeneity of class labels within a node. During training, the algorithm evaluates candidate feature thresholds and selects the split that minimizes the weighted Gini impurity across resulting child nodes.

### Q3: What is the main cause of overfitting in Decision Trees, and how is it mitigated?
> **Answer:** Overfitting occurs when a tree grows without depth limits (`max_depth=None`), creating overly specific rules that memorize training noise. It is mitigated by pre-pruning techniques such as setting `max_depth`, `min_samples_split`, and `min_samples_leaf`.

### Q4: How does a Decision Tree differ from Logistic Regression?
> **Answer:** Logistic Regression fits a smooth parametric sigmoid probability curve and requires feature scaling, whereas a Decision Tree constructs non-parametric orthogonal decision boundaries, handles non-linear relationships natively, and is invariant to monotonic feature transformations.

### Q5: How do you interpret a node in `sklearn.tree.plot_tree`?
> **Answer:** Each node displays:
> 1. The decision condition (e.g., `Study Hours <= 3.5`),
> 2. The Gini impurity of samples in that node,
> 3. The total sample count,
> 4. The distribution of samples per class (`value = [fail_count, pass_count]`), and
> 5. The majority class label (`class = Fail/Pass`).

---

## 🎉 Day 4 Completion & Achievements

```text
Day 4 Master Checklist:
├── Part 1: Decision Tree Basics & Intuition        ✅
├── Part 2: How Decision Trees Learn (Splits)       ✅
├── Part 3: Gini Impurity Manual Calculation        ✅
├── Part 4: Tree Depth & Overfitting Control        ✅
├── Part 5: Model Evaluation & Tree Visualization   ✅
└── Part 6: End-to-End Decision Tree Mini-Project   ✅
```

### 🏆 What You Have Mastered Today:
* The core architecture of tree-based algorithms (Root, Branches, Leaves).
* How trees automatically identify optimal splits from raw data.
* Manual and programmatic computation of Gini Impurity.
* Diagnosing and preventing overfitting using `max_depth`.
* Evaluating classification models using five essential metrics.
* Visualizing and auditing learned trees using `plot_tree()`.
* Building a full production-ready classification workflow in Python.

Congratulations on completing **Day 4**! 🚀