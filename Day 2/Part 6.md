# DAY 2 — PART 6: Mini ML Project & Day 2 Master Revision

> **Overview:** The final milestone for Day 2! In this part, we assemble all concepts—Data, Features, Targets, Train/Test Split, Linear Regression, Predictions, MSE, and MAE—into an end-to-end Python project predicting student exam scores. We conclude with a comprehensive 14-concept master review and top interview questions! 🚀🎓

---

## 📋 Table of Contents

1. [🎯 Project Goal: Student Exam Marks Predictor](#1️⃣-project-goal)
2. [Step 1: Dataset Preparation](#2️⃣-step-1--dataset-preparation)
3. [Step 2: Train / Test Split](#3️⃣-step-2--train--test-split)
4. [Step 3: Model Instantiation](#4️⃣-step-3--model-instantiation)
5. [Step 4: Model Training with `fit()`](#5️⃣-step-4--model-training-with-fit)
6. [Step 5: Generating Test Predictions](#6️⃣-step-5--generating-test-predictions)
7. [Step 6: Model Evaluation (MSE & MAE)](#7️⃣-step-6--model-evaluation-mse--mae)
8. [Step 7: Real-World Inference for New Students](#8️⃣-step-7--real-world-inference-for-new-students)
9. [💻 Complete End-to-End Executable Script](#9️⃣-complete-end-to-end-executable-script)
10. [🧠 The 7-Step Machine Learning Lifecycle](#-the-7-step-machine-learning-lifecycle)
11. [🔥 Day 2 Master Revision: 14 Core Concepts](#-day-2-master-revision-14-core-concepts)
12. [🎯 Day 2 Interview Test: Top 10 Questions](#-day-2-interview-test-top-10-questions)

---

## 1️⃣ Project Goal

We are building a practical regression model to answer:
> **Question:** Given the number of hours a student studies, what is their expected exam mark?

```text
Study Hours  ──>  [ Trained ML Model ]  ──>  Predicted Exam Marks
     6       ──>     (Linear Reg)       ──>     ~72.8 Marks
```

Because marks represent a continuous numerical value, we use the **Linear Regression** algorithm.

---

## 2️⃣ Step 1 — Dataset Preparation

Our historical training dataset of student records:

```python
# Features: Hours Studied (2D array, shape: 8 x 1)
x = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Target: Exam Marks (1D list)
y = [35, 42, 50, 58, 65, 72, 80, 86]
```

### Dataset Table:
| Student # | Study Hours ($x$) | Actual Marks ($y$) |
| :---: | :---: | :---: |
| 1 | 1 | 35 |
| 2 | 2 | 42 |
| 3 | 3 | 50 |
| 4 | 4 | 58 |
| 5 | 5 | 65 |
| 6 | 6 | 72 |
| 7 | 7 | 80 |
| 8 | 8 | 86 |

* **$x$** $\longrightarrow$ **Feature / Input** (`Hours Studied`)
* **$y$** $\longrightarrow$ **Target / Ground Truth** (`Marks`)

---

## 3️⃣ Step 2 — Train / Test Split

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
```

```text
               Original Data (8 samples)
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    Train Set (6 samples)      Test Set (2 samples)
    (Model learns patterns)    (Unbiased evaluation)
```

---

## 4️⃣ Step 3 — Model Instantiation

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

The model object is initialized. At this point, it has not seen any data and has learned no parameters.

---

## 5️⃣ Step 4 — Model Training with `fit()`

```python
model.fit(x_train, y_train)
```

* 🔥 **Golden Rule:** `fit()` = **LEARN**
* The algorithm finds the optimal slope ($m$) and intercept ($b$) that best describes the linear relationship:
  $$\text{Marks} = (m \times \text{Hours}) + b$$

---

## 6️⃣ Step 5 — Generating Test Predictions

```python
prediction = model.predict(x_test)
```

The trained model computes predictions for the unseen test features:

```text
x_test (Unseen Hours: [3, 2])  ──>  [ Trained Model ]  ──>  Predictions: [49.77, 42.43]
```

---

## 7️⃣ Step 6 — Model Evaluation (MSE & MAE)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
```

* **MSE:** Measures average squared differences; heavily penalizes large errors.
* **MAE:** Measures average absolute difference in original mark units.
* **Objective:** Smaller values indicate higher model precision.

---

## 8️⃣ Step 7 — Real-World Inference for New Students

Now we use the trained model for its true purpose: predicting outcomes for new students!

```python
new_student_hours = [[9]]
new_prediction = model.predict(new_student_hours)

print(f"Predicted Marks for 9 Hours: {new_prediction[0]:.2f}")
```

**Output:**
```text
Predicted Marks for 9 Hours: 94.21
```

---

## 9️⃣ Complete End-to-End Executable Script

Here is the entire project in a single, self-contained Python script:

```python
"""
Day 2 Capstone Project: Student Exam Mark Predictor
Algorithm: Linear Regression
Evaluation Metrics: Mean Squared Error (MSE) & Mean Absolute Error (MAE)
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    print("=" * 60)
    print("🎓 STUDENT EXAM MARK REGRESSION SYSTEM")
    print("=" * 60)

    # 1. Dataset Preparation
    x = [[1], [2], [3], [4], [5], [6], [7], [8]]
    y = [35, 42, 50, 58, 65, 72, 80, 86]

    # 2. Train / Test Split (75% Train, 25% Test)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    # 3. Model Creation & Training
    model = LinearRegression()
    model.fit(x_train, y_train)

    # 4. Model Parameters Inspection
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"\nModel Equation: Marks = ({slope:.2f} * Hours) + {intercept:.2f}")

    # 5. Evaluate on Unseen Test Data
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Actual Test Marks (y_test) : {y_test}")
    print(f"Predicted Marks (y_pred)   : {[round(p, 2) for p in predictions]}")
    print(f"Mean Squared Error (MSE)   : {mse:.4f}")
    print(f"Mean Absolute Error (MAE)  : {mae:.4f}")

    # 6. Real-World Inferences
    print("\n--- Real-World Predictions ---")
    for hours in [8, 9, 10]:
        pred = model.predict([[hours]])[0]
        print(f"Study Time: {hours} Hours  ──>  Estimated Marks: {pred:.1f} / 100")

    print("\n" + "=" * 60)
    print("✨ Day 2 Project Execution Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 🧠 The 7-Step Machine Learning Lifecycle

```text
1. DATA           (Gather historical examples)
    │
2. SPLIT          (Partition into Train and Test subsets)
    │
3. CREATE MODEL   (Instantiate algorithm object)
    │
4. TRAIN          (model.fit(x_train, y_train))
    │
5. PREDICT        (model.predict(x_test))
    │
6. EVALUATE       (Measure errors via MSE & MAE)
    │
7. INFERENCE      (Apply trained model to new inputs)
```

> 🔥 **Golden Memory Mantra:**  
> **$\text{Data} \longrightarrow \text{Split} \longrightarrow \text{Fit} \longrightarrow \text{Predict} \longrightarrow \text{Evaluate}$**

---

## 🔥 Day 2 Master Revision: 14 Core Concepts

| # | Concept | Definition / Core Meaning |
| :-: | :--- | :--- |
| **1** | **Data** | Collected historical examples and observations used for training. |
| **2** | **Feature** | The independent input variable ($x$) provided to the model. |
| **3** | **Target** | The dependent output ground truth ($y$) the model learns to predict. |
| **4** | **Training** | The iterative process where the algorithm learns patterns from data. |
| **5** | **Model** | A mathematical function mapping inputs to predicted outputs. |
| **6** | **`fit()`** | Scikit-learn method that trains the model on data. |
| **7** | **`predict()`** | Scikit-learn method that generates predictions for new inputs. |
| **8** | **Linear Regression** | Supervised algorithm that predicts continuous values using a straight line ($y = mx + b$). |
| **9** | **Error** | Difference between an individual actual value and prediction ($y - \hat{y}$). |
| **10** | **Loss Function** | An aggregate mathematical formula measuring total model error across a dataset. |
| **11** | **MSE** | Mean Squared Error; average of squared errors, penalizing large outliers. |
| **12** | **MAE** | Mean Absolute Error; average absolute error expressed in original units. |
| **13** | **Train/Test Split** | Dividing data into training (for learning) and testing (for unbiased evaluation). |
| **14** | **`random_state`** | Seed parameter locking the random generator for reproducible splits. |

---

## 🎯 Day 2 Interview Test: Top 10 Questions

### 1. What is a feature in Machine Learning?
> **Answer:** An independent input variable used by the model to compute predictions.

### 2. What is a target?
> **Answer:** The dependent ground-truth variable that the model is trained to predict.

### 3. What is the difference between `fit()` and `predict()`?
> **Answer:** `fit()` computes model weights from training data; `predict()` applies those weights to new inputs to generate predictions.

### 4. Why do we split datasets into train and test sets?
> **Answer:** To evaluate how well a model generalizes to unseen data and detect overfitting.

### 5. What is Linear Regression?
> **Answer:** A supervised learning algorithm for predicting continuous numeric values by fitting an optimal linear equation ($y = mx + b$).

### 6. What is Mean Squared Error (MSE)?
> **Answer:** An evaluation metric that computes the average of squared differences between actual and predicted values.

### 7. What is the key difference between MAE and MSE?
> **Answer:** MSE squares errors, heavily penalizing large outliers and changing the unit of measure. MAE uses absolute values, preserving the original units and offering intuitive interpretability.

### 8. What occurs when calling `model.fit(x_train, y_train)` in Linear Regression?
> **Answer:** The algorithm determines the optimal slope ($m$) and intercept ($b$) by minimizing the residual sum of squares.

### 9. Why shouldn't a model be trained using test data?
> **Answer:** Doing so causes data leakage, giving an artificially inflated performance estimate that fails to generalize in production.

### 10. Outline the standard 5-step ML workflow.
> **Answer:** Data Preparation $\rightarrow$ Train/Test Split $\rightarrow$ Model Training (`fit`) $\rightarrow$ Prediction (`predict`) $\rightarrow$ Performance Evaluation (Metrics).