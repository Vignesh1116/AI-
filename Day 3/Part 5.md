# DAY 3 — PART 5: Precision, Recall & F1-Score

> **Overview:** In Part 4, we learned the four outcomes of the Confusion Matrix: TP, TN, FP, and FN. Now, we use those values to compute the most critical classification metrics used in industry: **Precision**, **Recall**, and **F1-Score**. We will explore when to optimize each metric, how the harmonic mean works, and how to implement them in Python using Scikit-Learn! ⚖️🎯

---

## 📋 Table of Contents

1. [🚨 Why Accuracy Can Be Misleading (The Fraud Example)](#1️⃣-why-accuracy-can-be-misleading)
2. [🎯 Precision: Quality of Positive Predictions](#2️⃣-precision-quality-of-positive-predictions)
3. [🔍 Recall: Quantity of Actual Positives Caught](#3️⃣-recall-quantity-of-actual-positives-caught)
4. [⚖️ Precision vs Recall: The Real-World Tug-of-War](#4️⃣-precision-vs-recall-the-real-world-tug-of-war)
5. [🎼 F1-Score: Harmonic Mean Balance](#5️⃣-f1-score-harmonic-mean-balance)
6. [🐍 Scikit-Learn Python Implementation](#6️⃣-scikit-learn-python-implementation)
7. [🔢 Step-by-Step Worked Example](#7️⃣-step-by-step-worked-example)
8. [🧠 Core Memory Points & Quick Cheat Sheet](#-core-memory-points--quick-cheat-sheet)
9. [🎯 Top Interview Questions & Answers](#-top-interview-questions--answers)
10. [🗺️ Day 3 Progress Status](#-day-3-progress-status)

---

## 1️⃣ Why Accuracy Can Be Misleading

Imagine a fraud detection system monitoring **100 credit card transactions**:
* **95 Transactions:** Legitimate / Normal (Class 0)
* **5 Transactions:** Fraudulent (Class 1)

If a naive model predicts that **every single transaction is Legitimate**:
$$\text{Accuracy} = \frac{95 \text{ Correct}}{100 \text{ Total}} = 95\%$$

```text
Actual Fraud Cases Caught = 0 / 5 (0% Success!)
```

This model provides 95% accuracy while completely failing to protect the bank from fraud! To evaluate performance on the minority class properly, we rely on **Precision** and **Recall**.

---

## 2️⃣ Precision: Quality of Positive Predictions

> **Precision asks:** *"Out of all instances predicted as Positive, how many were actually Positive?"*

### 📐 Mathematical Formula:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{True Positives}}{\text{Total Predicted Positives}}$$

```text
                  All Predicted Positives (TP + FP)
               ┌──────────────────────┬──────────────────────┐
               │    TP (Correct)      │      FP (False Alarm)│
               └──────────────────────┴──────────────────────┘
                         ▲
                         │ Precision measures this fraction!
```

### 📧 Real-World Example: Spam Filter
* The model flags **100 emails** as Spam:
  * $\text{TP} = 80$ (Actually Spam ✅)
  * $\text{FP} = 20$ (Important personal emails incorrectly flagged ❌)

$$\text{Precision} = \frac{80}{80 + 20} = \frac{80}{100} = 0.80 \implies 80\%$$

> 💡 **Golden Rule:** When **False Positives (False Alarms)** are costly or harmful, prioritize **High Precision**!

---

## 3️⃣ Recall: Quantity of Actual Positives Caught

> **Recall (Sensitivity) asks:** *"Out of all actual Positive cases that exist, how many did the model find?"*

### 📐 Mathematical Formula:
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{True Positives}}{\text{Total Actual Positives}}$$

```text
                    All Actual Positives (TP + FN)
               ┌──────────────────────┬──────────────────────┐
               │    TP (Caught)       │      FN (Missed)     │
               └──────────────────────┴──────────────────────┘
                         ▲
                         │ Recall measures this fraction!
```

### 🏥 Real-World Example: Medical Screening
* A total of **90 patients** actually have a condition:
  * $\text{TP} = 80$ (Correctly detected ✅)
  * $\text{FN} = 10$ (Missed by the model 🚨)

$$\text{Recall} = \frac{80}{80 + 10} = \frac{80}{90} \approx 0.8889 \implies 88.89\%$$

> 💡 **Golden Rule:** When **False Negatives (Missed Cases)** are dangerous or unacceptable, prioritize **High Recall**!

---

## 4️⃣ Precision vs Recall: The Real-World Tug-of-War

| Metric | Core Focus | When to Prioritize? | Consequence of Failure |
| :--- | :--- | :--- | :--- |
| **Precision** | Quality of positive predictions | Spam filters, Loan approvals, Recommendation systems | **FP is costly:** Important legitimate email sent to spam. |
| **Recall** | Quantity of actual positives caught | Medical diagnosis, Fraud detection, Security screening | **FN is costly:** Sick patient sent home undiagnosed. |

```text
                   PRECISION                          RECALL
            "Quality of Predictions"          "Completeness of Search"
           Minimizes False Positives          Minimizes False Negatives
```

---

## 5️⃣ F1-Score: Harmonic Mean Balance

Often, improving Precision lowers Recall, and vice versa. When you want a single score that balances both metrics, use the **F1-Score**.

### 📐 Formula (Harmonic Mean):
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

> ❓ **Why Harmonic Mean instead of an Arithmetic Average?**  
> An arithmetic average does not penalize extreme imbalances:  
> If $\text{Precision} = 100\%$ and $\text{Recall} = 0\%$:  
> * Arithmetic Mean: $\frac{1.0 + 0.0}{2} = 50\%$ (Misleading!)  
> * **Harmonic Mean ($F_1$):** $2 \times \frac{1 \times 0}{1 + 0} = \mathbf{0\%}$ (Accurately reflects failure!)

---

## 6️⃣ Scikit-Learn Python Implementation

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Ground Truth and Predictions
actual = [1, 1, 1, 0, 0, 0]
prediction = [1, 1, 0, 1, 0, 0]

# Compute individual scores
precision = precision_score(actual, prediction)
recall = recall_score(actual, prediction)
f1 = f1_score(actual, prediction)

print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")

# Detailed summary report
print("\n--- Detailed Classification Report ---")
print(classification_report(actual, prediction, target_names=["Negative (0)", "Positive (1)"]))
```

### Output:
```text
Precision : 0.6667 (66.67%)
Recall    : 0.6667 (66.67%)
F1 Score  : 0.6667 (66.67%)

--- Detailed Classification Report ---
              precision    recall  f1-score   support

Negative (0)       0.67      0.67      0.67         3
Positive (1)       0.67      0.67      0.67         3

    accuracy                           0.67         6
   macro avg       0.67      0.67      0.67         6
weighted avg       0.67      0.67      0.67         6
```

---

## 7️⃣ Step-by-Step Worked Example

Let's manually compute each metric for the sample above:

* $\text{Actual} = [1, 1, 1, 0, 0, 0]$
* $\text{Prediction} = [1, 1, 0, 1, 0, 0]$

### Manual Classification Table:
| Sample # | Actual ($y$) | Predicted ($\hat{y}$) | Category | Meaning |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 1 | 1 | **TP** | Correctly predicted Positive ✅ |
| 2 | 1 | 1 | **TP** | Correctly predicted Positive ✅ |
| 3 | 1 | 0 | **FN** | Positive missed as Negative ❌ |
| 4 | 0 | 1 | **FP** | Negative predicted as Positive ❌ |
| 5 | 0 | 0 | **TN** | Correctly predicted Negative ✅ |
| 6 | 0 | 0 | **TN** | Correctly predicted Negative ✅ |

### Confusion Matrix Counts:
$$\text{TP} = 2, \quad \text{TN} = 2, \quad \text{FP} = 1, \quad \text{FN} = 1$$

### Calculations:
1. **Precision:**  
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{2}{2 + 1} = \frac{2}{3} \approx 66.67\%$$

2. **Recall:**  
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{2}{2 + 1} = \frac{2}{3} \approx 66.67\%$$

3. **F1-Score:**  
   $$F_1 = 2 \times \frac{0.6667 \times 0.6667}{0.6667 + 0.6667} \approx 66.67\%$$

---

## 🧠 Core Memory Points & Quick Cheat Sheet

```text
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Precision ──> Out of all predicted Positives, how many  │
│                  are correct?                               │
│                  Formula: TP / (TP + FP)                    │
├─────────────────────────────────────────────────────────────┤
│ 🔍 Recall    ──> Out of all actual Positives, how many did  │
│                  we detect?                                 │
│                  Formula: TP / (TP + FN)                    │
├─────────────────────────────────────────────────────────────┤
│ ⚖️ F1 Score  ──> Harmonic mean balancing Precision & Recall │
│                  Formula: 2 * (P * R) / (P + R)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Top Interview Questions & Answers

### Q1. What is Precision?
> **Answer:**  
> Precision is the ratio of correctly predicted positive observations to total predicted positive observations:
> $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

### Q2. What is Recall?
> **Answer:**  
> Recall (also known as Sensitivity) is the ratio of correctly predicted positive observations to all observations in the actual positive class:
> $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### Q3. Why do we use the harmonic mean for F1-Score instead of the arithmetic mean?
> **Answer:**  
> The harmonic mean heavily penalizes extreme values. If either Precision or Recall is close to zero, the F1-Score drops close to zero, ensuring that a high score requires both metrics to perform reasonably well.

### Q4. When would you prioritize Recall over Precision?
> **Answer:**  
> When missing a positive case carries a severe cost or safety hazard, such as cancer detection, airport security weapon scans, or credit card fraud detection.

---

## 🗺️ Day 3 Progress Status

```text
Part 1: Classification Basics             [COMPLETED] ✅
Part 2: Logistic Regression Deep Dive     [COMPLETED] ✅
Part 3: Train/Test Split & Accuracy       [COMPLETED] ✅
Part 4: Confusion Matrix Deep Dive        [COMPLETED] ✅
Part 5: Precision, Recall & F1 Score      [COMPLETED] ✅
Part 6: End-to-End Classification Project [FINAL STEP] 🚀
```