# DAY 3 — PART 6: Mini ML Project — Student Pass / Fail Classifier

> **Overview:** The final milestone for Day 3! In this part, we assemble everything we have learned—Classification, Logistic Regression, Train/Test Split, Accuracy, Confusion Matrix, Precision, Recall, F1-Score, and Probability inference—into a complete, end-to-end Python machine learning project! 🚀🎓

---

## 📋 Table of Contents

1. [🎯 Project Objective & Architecture](#1️⃣-project-objective--architecture)
2. [📊 Step 1: Problem Definition & Dataset](#2️⃣-step-1-problem-definition--dataset)
3. [📦 Step 2: Import Scikit-Learn Modules](#3️⃣-step-2-import-scikit-learn-modules)
4. [✂️ Step 3: Train / Test Split](#4️⃣-step-3-train--test-split)
5. [🤖 Step 4: Model Instantiation & Training](#5️⃣-step-4-model-instantiation--training)
6. [🔮 Step 5: Test Predictions & Comparison](#6️⃣-step-5-test-predictions--comparison)
7. [📈 Step 6: Full Evaluation Metrics Suite](#7️⃣-step-6-full-evaluation-metrics-suite)
8. [🚀 Step 7: Real-World Prediction for New Students](#8️⃣-step-7-real-world-prediction-for-new-students)
9. [💻 Complete End-to-End Executable Script](#9️⃣-complete-end-to-end-executable-script)
10. [🗺️ The Complete Day 3 Unified Flow](#-the-complete-day-3-unified-flow)
11. [🎯 Day 3 Master Interview Revision](#-day-3-master-interview-revision)
12. [🏆 Day 3 Milestone Summary](#-day-3-milestone-summary)

---

## 1️⃣ Project Objective & Architecture

* **Goal:** Predict whether a student will **Pass (1)** or **Fail (0)** an exam based on the number of hours they studied using a binary classification model.
* **Architecture Flow:**

```text
       [ Raw Dataset: Hours vs Result ]
                      │
                      ▼
            [ train_test_split() ]
             ├── 80% Training Data
             └── 20% Testing Data
                      │
                      ▼
         [ LogisticRegression.fit() ]
                      │
                      ▼
        [ model.predict(X_test) ]
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
[ Evaluation Suite ]          [ Real-World Inference ]
├── Accuracy                  ├── predict([[6 hrs]]) ──> Class
├── Confusion Matrix          └── predict_proba([[6 hrs]]) ──> Probabilities
├── Precision & Recall
└── F1-Score
```

---

## 2️⃣ Step 1: Problem Definition & Dataset

* **Feature ($X$):** Study Hours (Input Matrix, Shape `(10, 1)`)
* **Target ($y$):** Exam Result (0 = Fail, 1 = Pass)

### Dataset Table:
| Student # | Study Hours ($X$) | Actual Label ($y$) | Meaning |
| :---: | :---: | :---: | :---: |
| 1 | `[1]` | `0` | Fail |
| 2 | `[2]` | `0` | Fail |
| 3 | `[3]` | `0` | Fail |
| 4 | `[4]` | `1` | Pass |
| 5 | `[5]` | `1` | Pass |
| 6 | `[6]` | `1` | Pass |
| 7 | `[7]` | `1` | Pass |
| 8 | `[8]` | `1` | Pass |
| 9 | `[9]` | `1` | Pass |
| 10 | `[10]` | `1` | Pass |

```python
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

---

## 3️⃣ Step 2: Import Scikit-Learn Modules

We import the required modules from Scikit-Learn:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
```

---

## 4️⃣ Step 3: Train / Test Split

We allocate **80%** of the data for training and **20%** for unbiased evaluation:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training samples : {len(x_train)}")
print(f"Testing samples  : {len(x_test)}")
```

---

## 5️⃣ Step 4: Model Instantiation & Training

```python
# 1. Create Model Object
model = LogisticRegression()

# 2. Train on Training Data
model.fit(x_train, y_train)
```

> 💡 **What Happens During Training?**  
> The model finds the best weights ($m$) and intercept ($b$) to map study hours through the Sigmoid function:  
> $$P(\text{Pass}) = \frac{1}{1 + e^{-(mx + b)}}$$

---

## 6️⃣ Step 5: Test Predictions & Comparison

```python
prediction = model.predict(x_test)

print("Actual Test Labels    :", y_test)
print("Model Predicted Labels:", prediction)
```

### Sample Output:
```text
Actual Test Labels    : [0, 1]
Model Predicted Labels: [0, 1]
```
Both test samples were correctly classified by the model!

---

## 7️⃣ Step 6: Full Evaluation Metrics Suite

```python
# Accuracy
acc = accuracy_score(y_test, prediction)
print(f"Accuracy         : {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix :")
print(cm)

# Precision, Recall, F1-Score
precision = precision_score(y_test, prediction)
recall    = recall_score(y_test, prediction)
f1        = f1_score(y_test, prediction)

print(f"Precision Score  : {precision * 100:.2f}%")
print(f"Recall Score     : {recall * 100:.2f}%")
print(f"F1 Score         : {f1 * 100:.2f}%")
```

### Metrics Output Breakdown:
```text
Accuracy         : 100.00%
Confusion Matrix :
[[1 0]
 [0 1]]
Precision Score  : 100.00%
Recall Score     : 100.00%
F1 Score         : 100.00%
```

> ⚠️ **Note on Small Datasets:**  
> Because our sample dataset is small and easily separable, the model achieves 100% accuracy. On large, noisy real-world datasets, scores will be lower and more varied.

---

## 8️⃣ Step 7: Real-World Prediction for New Students

Now we use the trained model to predict outcomes for new students:

### Case: A student studying 6 hours
```python
new_student_hours = [[6]]

predicted_class = model.predict(new_student_hours)
predicted_probs = model.predict_proba(new_student_hours)

status = "Pass" if predicted_class[0] == 1 else "Fail"

print(f"Study Time   : 6 Hours")
print(f"Prediction   : Class {predicted_class[0]} ({status})")
print(f"Probabilities: Fail = {predicted_probs[0][0]*100:.1f}%, Pass = {predicted_probs[0][1]*100:.1f}%")
```

### Output:
```text
Study Time   : 6 Hours
Prediction   : Class 1 (Pass)
Probabilities: Fail = 10.4%, Pass = 89.6%
```

---

## 9️⃣ Complete End-to-End Executable Script

Here is the complete Python script combining all steps:

```python
"""
Day 3 Capstone Project: Student Pass/Fail Classification
Algorithm: Logistic Regression
Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1 Score
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def main():
    print("=" * 60)
    print("🎓 STUDENT PASS/FAIL CLASSIFICATION SYSTEM")
    print("=" * 60)

    # 1. Dataset
    x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # 2. Train / Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # 3. Model Creation & Training
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # 4. Evaluation on Unseen Test Data
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Actual Labels    : {y_test}")
    print(f"Predicted Labels : {predictions}")
    print(f"Accuracy         : {accuracy * 100:.2f}%")
    print(f"Precision        : {precision * 100:.2f}%")
    print(f"Recall           : {recall * 100:.2f}%")
    print(f"F1-Score         : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)

    # 5. New Student Inferences
    print("\n--- Real-World Inferences ---")
    test_cases = [2, 4, 7]
    for hours in test_cases:
        pred_label = model.predict([[hours]])[0]
        prob = model.predict_proba([[hours]])[0]
        label_text = "PASS ✅" if pred_label == 1 else "FAIL ❌"
        print(f"Study: {hours} hrs | Predicted: {label_text} | Conf: P(Fail)={prob[0]*100:.1f}%, P(Pass)={prob[1]*100:.1f}%")

    print("\n" + "=" * 60)
    print("✨ Project Execution Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 🗺️ The Complete Day 3 Unified Flow

```text
                           MACHINE LEARNING
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
            Regression                       Classification
         (Continuous Mark)                  (Pass / Fail Label)
                                                   │
                                                   ▼
                                         Binary Classification
                                           (2 Classes: 0 or 1)
                                                   │
                                                   ▼
                                          Logistic Regression
                                           (Sigmoid Function)
                                                   │
                                                   ▼
                                           train_test_split()
                                         (Train 80% / Test 20%)
                                                   │
                                                   ▼
                                             model.fit()
                                                   │
                                                   ▼
                                            model.predict()
                                                   │
                                ┌──────────────────┴──────────────────┐
                                ▼                                     ▼
                        Diagnostic Metrics                      Confidence Scores
                        ├── Accuracy                           └── predict_proba()
                        ├── Confusion Matrix (TP, TN, FP, FN)
                        ├── Precision (Positive Quality)
                        ├── Recall (Positive Quantity)
                        └── F1-Score (Harmonic Balance)
```

---

## 🎯 Day 3 Master Interview Revision

### 1. What is classification in Machine Learning?
> **Answer:**  
> Classification is a supervised learning technique that assigns input data into predefined discrete categories or classes (e.g., spam vs non-spam).

### 2. What algorithm did we use for binary classification?
> **Answer:**  
> Logistic Regression. It predicts the probability of a categorical outcome by passing linear values through the Sigmoid activation function.

### 3. What is the difference between `predict()` and `predict_proba()`?
> **Answer:**  
> `predict()` outputs discrete class labels based on a decision threshold, while `predict_proba()` outputs the raw probabilities for each class.

### 4. Why is Accuracy alone not enough for evaluation?
> **Answer:**  
> On imbalanced datasets, a model that simply predicts the majority class can yield high accuracy while completely failing to detect the minority class.

### 5. What is a Confusion Matrix?
> **Answer:**  
> A table evaluating model performance by displaying counts of True Positives, True Negatives, False Positives, and False Negatives.

### 6. What is False Positive vs False Negative?
> **Answer:**  
> * **FP (Type I error):** Model falsely predicted positive when the actual class was negative (False alarm).  
> * **FN (Type II error):** Model falsely predicted negative when the actual class was positive (Dangerous miss).

### 7. What is Precision?
> **Answer:**  
> The proportion of predicted positives that were actually positive: $\frac{\text{TP}}{\text{TP} + \text{FP}}$.

### 8. What is Recall?
> **Answer:**  
> The proportion of actual positives that were correctly caught: $\frac{\text{TP}}{\text{TP} + \text{FN}}$.

### 9. What is F1-Score?
> **Answer:**  
> The harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

### 10. Why use Train/Test Split?
> **Answer:**  
> To train the model on one portion of the data and evaluate its ability to generalize on unseen data, preventing overfitting and data leakage.

---

## 🏆 Day 3 Milestone Summary

| Milestone Concept | Core Takeaway | Day 3 Status |
| :--- | :--- | :---: |
| **Regression vs Classification** | Numbers vs Discrete Categories | ✅ **Mastered** |
| **Logistic Regression** | Sigmoid curve squashing to probabilities $[0, 1]$ | ✅ **Mastered** |
| **Train / Test Split** | Prevent data leakage & test generalization | ✅ **Mastered** |
| **Confusion Matrix** | Diagnostic matrix of TP, TN, FP, FN | ✅ **Mastered** |
| **Precision & Recall** | Prediction quality vs Detection completeness | ✅ **Mastered** |
| **F1-Score** | Harmonic mean balancing Precision & Recall | ✅ **Mastered** |
| **Mini ML Project** | End-to-end model training, evaluation & inference | ✅ **Completed** |

🎉 **Congratulations! Day 3 is 100% complete in clear English with full theoretical mastery and practical code!**