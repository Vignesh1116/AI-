# DAY 2 — PART 4: Train / Test Split

> **Overview:** In Machine Learning, evaluating a model on the same data it was trained on is like giving students the exact exam questions before test day. To measure genuine learning and generalization, we must split our dataset into **Training Data** and **Testing Data**. In this part, we explore the train/test split ratio, the Scikit-learn implementation, and why setting `random_state` matters! 🧪📊

---

## 📋 Table of Contents

1. [Real-World Intuition: The Exam Analogy 👶](#1-real-world-intuition-the-exam-analogy)
2. [Why Do We Split Datasets?](#2-why-do-we-split-datasets)
3. [Training Data vs Testing Data](#3-training-data-vs-testing-data)
4. [Standard Train / Test Split Ratios](#4-standard-train--test-split-ratios)
5. [How to Split Data in Python (`train_test_split`)](#5-how-to-split-data-in-python-train_test_split)
6. [Understanding Split Function Parameters](#6-understanding-split-function-parameters)
7. [Why Do We Get Four Separate Variables?](#7-why-do-we-get-four-separate-variables)
8. [Complete End-to-End Code](#8-complete-end-to-end-code)
9. [The Full ML Workflow Diagram 🔥](#9-the-full-ml-workflow-diagram)
10. [What is `random_state=42`?](#10-what-is-random_state42)
11. [⚠️ The Golden Rule: Avoid Data Leakage](#11-the-golden-rule-avoid-data-leakage)
12. [🎯 Top Interview Questions & Answers](#12-top-interview-questions--answers)

---

## 1. Real-World Intuition: The Exam Analogy 👶

Imagine preparing for a critical final examination:

```text
100 Practice Questions  ──>  Study, Practice, & Learn Patterns
                                     │
                                     ▼
New Exam Questions      ──>  Test Real Knowledge (Unseen)
```

1. Your instructor gives you 100 sample problems to study.
2. You practice solving them until you understand the underlying concepts.
3. On exam day, the instructor tests you with **brand-new questions** you have never seen before.
4. If you score well on the new questions, you truly understand the subject!

In Machine Learning, we follow the exact same methodology:

```text
               Original Dataset
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      Training Set          Testing Set
      (Learn from it)       (Evaluate on it)
```

---

## 2. Why Do We Split Datasets?

Suppose our entire dataset consists of 8 students:

| Hours Studied ($x$) | Marks ($y$) |
| :---: | :---: |
| 1 | 35 |
| 2 | 42 |
| 3 | 50 |
| 4 | 58 |
| 5 | 65 |
| 6 | 72 |
| 7 | 80 |
| 8 | 86 |

If we train our model on all 8 students:
```python
model.fit(x, y)
```
and then test the model on the exact same 8 students, a high score only tells us that the model memorized the training examples.

The primary question in machine learning is:
> **"How well does the model perform on brand-new, unseen data?"**

By keeping a subset of data hidden during training, we can fairly measure real-world performance!

---

## 3. Training Data vs Testing Data

### 🏋️ Training Data
* The subset of data used by the algorithm to **learn patterns and calculate model weights ($m$ and $b$)**.
* **Example:** Hours 1 through 6 $\implies$ `model.fit(x_train, y_train)`.

### 🧪 Testing Data
* The subset held back to **evaluate generalization** after training finishes.
* **Example:** Hours 7 and 8 $\implies$ `model.predict(x_test)`.
* Model predictions are then compared against ground-truth labels `y_test`.

---

## 4. Standard Train / Test Split Ratios

We divide the dataset into two distinct partitions:

```text
                     FULL DATASET
                          │
                  train_test_split()
                     ↙          ↘
            Training Set       Testing Set
            (70% - 80%)        (20% - 30%)
                 │                  │
               Learn             Evaluate
```

### Common Split Configurations:
* **80% Training / 20% Testing** *(Most common standard)*
* **75% Training / 25% Testing** *(Common for small-to-medium datasets)*
* **70% Training / 30% Testing** *(Standard alternative)*

---

## 5. How to Split Data in Python (`train_test_split`)

Scikit-Learn provides a built-in utility function:

```python
from sklearn.model_selection import train_test_split

x = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [35, 42, 50, 58, 65, 72, 80, 86]

# Split data: 75% train, 25% test
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.25, 
    random_state=42
)
```

---

## 6. Understanding Split Function Parameters

* **`x`**: Input features matrix (2D array).
* **`y`**: Target output vector (1D list/array).
* **`test_size=0.25`**:
  * 25% of the samples are reserved for the test set.
  * 75% are kept for training.
  * For our 8-sample dataset: $8 \times 0.25 = \mathbf{2\text{ test samples}}$ and $\mathbf{6\text{ training samples}}$.
* **`random_state=42`**:
  * Fixes the random shuffling seed so the split produces the exact same subsets across every run.

---

## 7. Why Do We Get Four Separate Variables?

Remember the core machine learning convention:
$$\mathbf{X} = \text{Features (Inputs)}, \quad \mathbf{y} = \text{Targets (Outputs)}$$

Splitting both inputs and outputs yields 4 distinct variables:

| Variable | Type | Contents | Purpose |
| :--- | :--- | :--- | :--- |
| **`x_train`** | Input | Study hours for 6 students | Inputs given to `model.fit()` |
| **`y_train`** | Output | Actual marks for 6 students | Correct answers used during training |
| **`x_test`** | Input | Study hours for 2 test students | Unseen inputs given to `model.predict()` |
| **`y_test`** | Output | Actual marks for 2 test students | Ground truth used to evaluate predictions |

---

## 8. Complete End-to-End Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Full Dataset
x = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [35, 42, 50, 58, 65, 72, 80, 86]

# 2. Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# 3. Train Model Using Training Data Only
model = LinearRegression()
model.fit(x_train, y_train)

# 4. Generate Predictions on Test Features
predictions = model.predict(x_test)

# 5. Compare Predictions with Ground Truth
print("Actual Test Marks (y_test) :", y_test)
print("Predicted Marks            :", predictions)
```

### Output:
```text
Actual Test Marks (y_test) : [50, 42]
Predicted Marks            : [49.77142857, 42.42857143]
```

---

## 9. The Full ML Workflow Diagram 🔥

```text
       Raw Dataset (x, y)
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
                   [ Evaluate against y_test ]
```

---

## 10. What is `random_state=42`?

When splitting data, Scikit-learn shuffles samples randomly to ensure representative distributions.

* Without `random_state`, every execution produces a slightly different split.
* By setting `random_state=42` (or any constant integer), the pseudo-random generator is initialized with a fixed seed.
* This guarantees that your colleagues, mentors, or automated tests obtain the exact same train/test split.

---

## 11. ⚠️ The Golden Rule: Avoid Data Leakage

> 🚨 **Critical Rule of Machine Learning:**  
> **Never expose test data to the model during training!**

* Always call `model.fit(x_train, y_train)`.
* **Never** call `model.fit(x, y)` or include test data in the fit step before evaluation.
* Allowing test samples to influence training is known as **Data Leakage**, and it creates false illusions of high accuracy that fail in production.

---

## 12. 🎯 Top Interview Questions & Answers

### Q1. Why do we split datasets into train and test sets?
> **Answer:**  
> To evaluate how well a model generalizes to new, unseen data and to detect whether the model is overfitting on the training data.

### Q2. What is the standard split ratio?
> **Answer:**  
> Typically 80/20 or 75/25 for small to moderately sized datasets. For massive datasets with millions of rows, ratios like 95/5 or 98/2 are commonly used.

### Q3. What is Data Leakage?
> **Answer:**  
> Data leakage occurs when information from outside the training dataset (such as test labels or future observations) inadvertently leaks into the model training pipeline, resulting in unrealistically optimistic performance metrics that fail in real-world deployment.