# DAY 2 — PART 6: Mini ML Project & Day 2 Master Revision

> **Overview:** Idhu namma Day 2 final part! End-to-end Mini Project (Student Marks Prediction) build panni, Day 2 concepts ellam revise panni, final interview test attend pannuvom! 🚀

---

## 📋 Table of Contents

1. [🎯 Project Goal: Student Marks Prediction](#-project-goal)
2. [Step 1 — Dataset Preparation](#1️⃣-step-1--dataset)
3. [Step 2 — Train / Test Split](#2️⃣-step-2--split-the-data)
4. [Step 3 — Initialize Model](#3️⃣-step-3--create-model)
5. [Step 4 — Train Model (`fit`)](#4️⃣-step-4--train)
6. [Step 5 — Predict on Test Set](#5️⃣-step-5--predict-test-data)
7. [Step 6 — Model Evaluation (MSE & MAE)](#6️⃣-step-6--evaluate)
8. [Step 7 — Predict for a Brand New Student](#7️⃣-step-7--predict-a-new-student)
9. [🧩 Complete Project Code (One Shot)](#-complete-project-code)
10. [The 7-Step ML Lifecycle & Memory Mantra](#-ippo-code-a-understand-pannanum)
11. [🔥 Day 2 Complete Revision (14 Concepts)](#-day-2-complete-revision)
12. [🎯 Day 2 Interview Test (10 Questions)](#-day-2-interview-test)

---

## 🎯 Project Goal

Namma oru simple practical ML application build panna porom:
> **Question:** Student evlo hours study pannirukanga $\longrightarrow$ Avanga expected marks evlo?

```text
Study Hours  ──>  [ Trained ML Model ]  ──>  Predicted Marks
     6       ──>     (Linear Reg)       ──>     ~73 Marks
```

Indha problem continuous numerical value predict panradhala, namma **Linear Regression** algorithm use panrom.

---

## 1️⃣ Step 1 — Dataset

Namma historical student training data:

```python
# Features: Hours Studied (2D array)
x = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Target: Marks Obtained
y = [35, 42, 50, 58, 65, 72, 80, 86]
```

### Data Table:
| Study Hours ($x$) | Marks ($y$) |
| :---: | :---: |
| 1 | 35 |
| 2 | 42 |
| 3 | 50 |
| 4 | 58 |
| 5 | 65 |
| 6 | 72 |
| 7 | 80 |
| 8 | 86 |

* **$x$** $\longrightarrow$ **Feature / Input** (`Hours Studied`)
* **$y$** $\longrightarrow$ **Target / Output** (`Marks`)

---

## 2️⃣ Step 2 — Split the Data

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
```

### Why Split?
Model-ku ellame data-vaiyum training-la kuduthutta, model genuinely generalize aagi learn pannucha-nu verify panna mudiyadhu.

```text
               Original Data (8 samples)
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    Train Set (6 samples)      Test Set (2 samples)
    (Model learns patterns)    (Fair evaluation)
```

---

## 3️⃣ Step 3 — Create Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

* Ippo model object instantiate panniyachu.
* ⚠️ **Important:** Innum model training data-va paakkala, learn pannala.

---

## 4️⃣ Step 4 — Train

```python
model.fit(x_train, y_train)
```

* 🔥 **Very Important:** `fit()` = **LEARN**
* Model $x_{\text{train}}$ and $y_{\text{train}}$-la irukkura relationship-ah ($m$ slope and $b$ intercept) calculate panni learn pannum.
  $$\text{Study Hours} \uparrow \implies \text{Marks} \uparrow$$

---

## 5️⃣ Step 5 — Predict Test Data

```python
prediction = model.predict(x_test)
```

Ippo model test features-ku predictions calculate pannum:

```text
x_test (Unseen Hours)  ──>  [ Trained Model ]  ──>  Predictions
```

---

## 6️⃣ Step 6 — Evaluate

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

print("MSE:", mse)
print("MAE:", mae)
```

* **MSE (Mean Squared Error):** Large mistakes-ku heavy penalty kudukkum.
* **MAE (Mean Absolute Error):** Average absolute difference between actual & predicted values.
* **Goal:** Lower MSE & Lower MAE $\implies$ Better Model accuracy.

---

## 7️⃣ Step 7 — Predict a New Student

Suppose a new student studies **9 hours**:

```python
new_prediction = model.predict([[9]])
print("Predicted Marks for 9 Hours:", new_prediction)
```

> 💡 **This is the true purpose of ML:**  
> Historical data-la irundhu pattern learn pannitu, brand new unseen inputs-ku accurate predictions produce panradhu!

---

## 🧩 Complete Project Code

Entire project implementation in one clean, ready-to-run script:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Dataset
x = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [35, 42, 50, 58, 65, 72, 80, 86]

# 2. Split Data (75% Train, 25% Test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# 3. Create Model
model = LinearRegression()

# 4. Train Model
model.fit(x_train, y_train)

# 5. Predict on Test Set
prediction = model.predict(x_test)

# 6. Evaluate Performance
mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

print("--- Model Evaluation ---")
print("Test Actual (y_test): ", y_test)
print("Test Predicted:       ", prediction)
print("MSE:                  ", mse)
print("MAE:                  ", mae)

# 7. Real-World Inference (New Student - 9 Hours)
new_prediction = model.predict([[9]])
print("\n--- Real Inference ---")
print("Predicted Marks for 9 Hours:", new_prediction[0])
```

---

## 🧠 Ippo Code-a Understand Pannanum

Entire project-a simple 7 steps-ah mind-la vechuko:

```text
1. DATA           (Gather historical samples)
    │
2. SPLIT          (Divide into Train and Test)
    │
3. CREATE MODEL   (Instantiate LinearRegression)
    │
4. TRAIN          (model.fit() -> learn m and b)
    │
5. PREDICT        (model.predict(x_test))
    │
6. EVALUATE       (Calculate MSE and MAE)
    │
7. NEW PREDICTION (Inference on unseen inputs)
```

> 🔥 **Golden Memory Mantra:**  
> **$\text{Data} \longrightarrow \text{Split} \longrightarrow \text{Fit} \longrightarrow \text{Predict} \longrightarrow \text{Evaluate}$**

---

## 🔥 Day 2 Complete Revision

Quick reference revision for all 14 core concepts:

| # | Concept | Definition / Meaning |
| :-: | :--- | :--- |
| **1** | **Data** | Information or examples collected for learning patterns. |
| **2** | **Feature** | The input variable ($x$) used by the model for making predictions. |
| **3** | **Target** | The output ground-truth value ($y$) that the model tries to predict. |
| **4** | **Training** | The process where a model learns patterns and weights from data. |
| **5** | **Model** | A learned mathematical system that maps inputs to predicted outputs. |
| **6** | **`fit()`** | Method that trains the model using training data. |
| **7** | **`predict()`** | Method that uses the trained model to generate predictions for new data. |
| **8** | **Linear Regression** | Supervised learning algorithm that predicts continuous numerical values along a straight line ($y = mx + b$). |
| **9** | **Error** | Difference between an individual actual value and predicted value ($\text{Actual} - \text{Predicted}$). |
| **10** | **Loss Function** | Function measuring how wrong the model is across the dataset. |
| **11** | **MSE** | Mean Squared Error; average of squared errors, heavily penalizing large outliers. |
| **12** | **MAE** | Mean Absolute Error; average of absolute error values in original units. |
| **13** | **Train/Test Split** | Dividing data into training set (to learn) and testing set (for unbiased evaluation). |
| **14** | **`random_state`** | Seed parameter that locks the randomness for reproducible splits. |

---

## 🎯 DAY 2 INTERVIEW TEST

*(Interviewer unkitta kekura maari consider pannu. Answers paakama solve panna try pannu!)*

* **Q1.** What is a feature in Machine Learning?
* **Q2.** What is a target?
* **Q3.** What is the exact difference between `fit()` and `predict()`?
* **Q4.** Why do we split data into training and testing sets?
* **Q5.** What is Linear Regression?
* **Q6.** What is MSE?
* **Q7.** MAE vs MSE — what is the key difference?
* **Q8.** What happens internally when we call `model.fit(x_train, y_train)`?
* **Q9.** Why shouldn't we train the model using test data? *(Data Leakage)*
* **Q10.** Explain the basic end-to-end ML workflow from data to evaluation.