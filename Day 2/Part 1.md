# DAY 2 — Machine Learning Fundamentals

> **Overview:** Part by part-ah povom. Step-by-step understanding and your very first actual ML practical code!

---

## 📋 Today's Roadmap

1. [Data](#part-1--data)
2. [Features](#part-2--feature)
3. [Target / Label](#part-3--target--label)
4. [Training](#part-4--training)
5. [Model](#part-5--model)
6. [Prediction](#part-6--prediction)
7. [Complete ML Flow](#-complete-ml-flow)
8. [First Actual ML Code](#part-7--first-actual-ml-code)
9. [Key Takeaways & Interview Revision](#-two-words-strongly-remember)
10. [Your Hands-on Task](#-your-task)

---

## Part 1 — Data

ML-ku most important thing: **DATA**.

### Example Dataset: Students' Study Information

| Hours Studied | Marks |
| :---: | :---: |
| 1 | 35 |
| 2 | 42 |
| 3 | 50 |
| 4 | 58 |
| 5 | 65 |
| 6 | 72 |
| 7 | 80 |

Idhu namma data.

> 💡 **Definition:**  
> **Data** is simply information or examples collected for learning patterns.

---

## Part 2 — Feature

Indha table-la **Hours Studied** namma prediction panna use panra input.

> **Feature** = Input variable used by the ML model.

* **Here:** Feature = `Hours Studied`

### Multiple Features Example:
Later real-world projects-la multiple features irukkalam:
- Hours Studied
- Attendance
- Previous Marks
- Assignment Score

```text
Student Profile:
├── Hours Studied  = 5
├── Attendance     = 90%
└── Previous Marks = 65
(These are all features / inputs)
```

---

## Part 3 — Target / Label

Namma predict panna vendiya output: **Marks**.

> **Target (or Label)** = The output that the model is trained to predict.

* **Feature** $\longrightarrow$ `Hours Studied` (Input)
* **Target** $\longrightarrow$ `Marks` (Output)

> 💡 **Quick Rule:**  
> **INPUT** $\longrightarrow$ **FEATURE**  
> **OUTPUT** $\longrightarrow$ **TARGET**

---

## Part 4 — Training

Ippo namma data model-ku kudukkurom.

```text
Historical Data
      │
      ▼
Features + Target
      │
      ▼
   TRAINING
      │
      ▼
Model Learns Patterns
```

### Example Patterns:
* $1\text{ hour} \longrightarrow 35\text{ marks}$
* $2\text{ hours} \longrightarrow 42\text{ marks}$
* $3\text{ hours} \longrightarrow 50\text{ marks}$
* $4\text{ hours} \longrightarrow 58\text{ marks}$

Model indha examples-la irukkura relationship-ah learn panna try pannum.

> 💡 **Definition:**  
> **Training** is the process where a model learns patterns and relationships from training data.

---

## Part 5 — Model

Training mudinja apram namakku **Trained Model** kidaikkum.

> **Definition:**  
> A **Model** is a learned pattern system that uses input data to produce predictions or outputs.

```text
Hours Studied  ──>  [ Trained Model ]  ──>  Predicted Marks
```

---

## Part 6 — Prediction

Suppose new student: **Hours Studied = 8**.

Historical data-la 8 hours example illa. But model already pattern learn pannirukku.

```text
8 Hours (New Input)  ──>  [ Trained Model ]  ──>  Predicted Marks
```

*This process of estimating the unseen output is called **Prediction**.*

---

## 🔥 Complete ML Flow

*(Idha mind-la vechuko)*

```text
                 DATA
                  │
                  ▼
          Features + Target
                  │
                  ▼
              TRAINING
                  │
                  ▼
            LEARNED MODEL
                  │
                  ▼
              NEW INPUT
                  │
                  ▼
             PREDICTION
```

---

## Part 7 — First Actual ML Code

Ippo theory mattum illa, actual Machine Learning model build pannuvom!  
Install pannirundha **scikit-learn** use pannalaam.

```python
from sklearn.linear_model import LinearRegression

# Features / Inputs (2D Array: [samples, features])
x = [[1], [2], [3], [4], [5], [6], [7]]

# Target / Outputs
y = [35, 42, 50, 58, 65, 72, 80]

# 1. Create Model
model = LinearRegression()

# 2. Train Model
model.fit(x, y)

# 3. Predict for New Input (e.g., 9 hours)
prediction = model.predict([[9]])
print(prediction)
```

---

### 🔍 Code Explanation (Step-by-Step)

#### 1. Import
```python
from sklearn.linear_model import LinearRegression
```
* `LinearRegression` algorithm-ah import pannrom.

#### 2. Features (`x`)
```python
x = [[1], [2], [3], [4], [5], [6], [7]]
```
* Idhu feature / input (`Hours Studied`).
* **Why `[[1], [2]]` instead of `[1, 2]`?**  
  Scikit-learn expects features in a **2D structure**:  
  `[number_of_samples, number_of_features]`  
  Here: $7\text{ students} \times 1\text{ feature}$.

#### 3. Target (`y`)
```python
y = [35, 42, 50, 58, 65, 72, 80]
```
* Idhu target / output (`Marks`).

#### 4. Model Creation
```python
model = LinearRegression()
```
* Model object create pannrom (innum learn pannala).

#### 5. Training (`fit`)
```python
model.fit(x, y)
```
* 🔥 **Very important:** `fit()` means **Learn from the training data**.  
* Model $x \text{ (Hours)} \longrightarrow y \text{ (Marks)}$ relationship-ah learn pannum.

#### 6. Prediction (`predict`)
```python
prediction = model.predict([[9]])
print(prediction)
```
* Ippo new input: $9\text{ hours}$.
* Model learned pattern use panni marks predict pannum.
* **Output (approx):** `[94.6]` *(or `[89.5]` for 8 hours)* depending on data fit.

---

## 🧠 Two Words Strongly Remember

| Method | Role | Meaning |
| :--- | :--- | :--- |
| `fit()` | **LEARN** | Trains the model using given data |
| `predict()` | **PREDICT** | Uses the trained model to generate predictions for new data |

### 🎯 Interview Q&A

1. **What does `fit()` do?**  
   > *"The `fit()` method trains the model using the provided training data (features and target). It learns the underlying patterns or weights."*

2. **What does `predict()` do?**  
   > *"The `predict()` method uses the trained model to generate output predictions for new, unseen input data."*

---

## 🎯 Your Task

Ippo same code-la **8** instead of 9 use panni run pannu:

```python
# Predict marks for 8 hours of study
prediction = model.predict([[8]])
print(prediction)
```

Then output enna vandhudhu nu send pannu!