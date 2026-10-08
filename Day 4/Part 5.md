# DAY 4 — PART 5: Model Evaluation & Tree Visualization

> **Overview:** One of the greatest advantages of Decision Trees over "black box" models (like Neural Networks) is their **interpretability**. You do not have to guess how a Decision Tree made a prediction—you can plot the entire tree and visually follow every single decision rule! In this part, we review standard classification evaluation metrics (Accuracy, Confusion Matrix, Precision, Recall, and F1 Score) and master tree visualization using `plot_tree` from `scikit-learn`. 🌳

---

## 📋 Table of Contents

1. [📊 Review: Classification Evaluation Metrics](#1️⃣-review-classification-evaluation-metrics)
2. [👁️ Why Visualize a Decision Tree? (White-Box Models)](#2️⃣-why-visualize-a-decision-tree-white-box-models)
3. [🎨 Visualizing Trees with `plot_tree()`](#3️⃣-visualizing-trees-with-plot_tree)
4. [🔍 Anatomy of a Visualized Tree Node](#4️⃣-anatomy-of-a-visualized-tree-node)
5. [🎨 Why Use `filled=True`?](#5️⃣-why-use-filledtrue)
6. [💻 Complete Working Code: Train & Visualize](#6️⃣-complete-working-code-train--visualize)
7. [📜 Alternative: Text Representation with `export_text`](#7️⃣-alternative-text-representation-with-export_text)
8. [🧠 Core Memory Points](#-core-memory-points)
9. [💼 Top Interview Questions & Answers](#-top-interview-questions--answers)
10. [🗺️ Day 4 Learning Roadmap](#-day-4-learning-roadmap)

---

## 1️⃣ Review: Classification Evaluation Metrics

Just like Logistic Regression, we evaluate a Decision Tree classifier using our standard classification metrics from Day 3:

```text
                 Actual Positive (1)    Actual Negative (0)
                ┌─────────────────────┬─────────────────────┐
Predicted (1)   │ True Positive (TP)  │ False Positive (FP) │
                ├─────────────────────┼─────────────────────┤
Predicted (0)   │ False Negative (FN) │ True Negative (TN)  │
                └─────────────────────┴─────────────────────┘
```

| Metric | Formula | What it Measures |
| :--- | :---: | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall proportion of correct predictions across all samples. |
| **Precision** | $\frac{TP}{TP + FP}$ | Out of all samples predicted as positive, how many were actually positive? |
| **Recall** | $\frac{TP}{TP + FN}$ | Out of all actual positive cases, how many did the model successfully identify? |
| **F1 Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean that balances Precision and Recall. |
| **Confusion Matrix** | Table of $TP, FP, FN, TN$ | Direct breakdown of classification errors and successes. |

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
```

---

## 2️⃣ Why Visualize a Decision Tree? (White-Box Models)

Many machine learning algorithms (like Support Vector Machines and Deep Neural Networks) are **Black Box** models: they give you an answer, but you cannot easily inspect *why* that answer was produced.

A Decision Tree is a **White Box** (or glass-box) model:
* Every decision rule is completely visible.
* You can audit every threshold and condition.
* Highly valued in regulated industries like **Healthcare**, **Finance**, and **Credit Scoring** where decisions must be explained to stakeholders.

```text
[ Feature Values ] ──> Follow branches step-by-step ──> [ Exact Reason for Prediction ]
```

---

## 3️⃣ Visualizing Trees with `plot_tree()`

`scikit-learn` provides a built-in visualization function: `sklearn.tree.plot_tree`. Combined with `matplotlib.pyplot`, it renders the decision graph directly.

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 6))
plot_tree(
    model,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)
plt.show()
```

### Parameter Breakdown:

| Parameter | Purpose |
| :--- | :--- |
| `decision_tree` | The trained `DecisionTreeClassifier` or `DecisionTreeRegressor` instance. |
| `feature_names` | List of human-readable feature names (e.g., `["Study Hours"]`). |
| `class_names` | List of human-readable target labels in order of numeric class (e.g., `["Fail", "Pass"]`). |
| `filled=True` | Colors nodes according to their dominant class and impurity level. |
| `rounded=True` | Renders node boxes with rounded corners for aesthetic polish. |

---

## 4️⃣ Anatomy of a Visualized Tree Node

When you inspect an individual node produced by `plot_tree()`, it displays several distinct pieces of information:

```text
┌───────────────────────────────┐
│     Study Hours <= 3.5        │  <── Decision condition (threshold)
│         gini = 0.469          │  <── Gini Impurity of this node
│         samples = 8           │  <── Total training samples in this node
│       value = [3, 5]          │  <── Count per class: [3 Fail, 5 Pass]
│        class = Pass           │  <── Dominant class label (majority vote)
└───────────────────────────────┘
```

1. **Condition (`Study Hours <= 3.5`):** The question determining whether a sample moves left (`True`) or right (`False`).
2. **`gini`:** The impurity score calculated using $1 - \sum p_i^2$.
3. **`samples`:** The number of training samples currently residing in this node.
4. **`value = [3, 5]`:** Distribution of samples across classes (`3` for class 0 / Fail, `5` for class 1 / Pass).
5. **`class = Pass`:** The predicted class if the tree terminated at this node (determined by the majority count in `value`).

---

## 5️⃣ Why Use `filled=True`?

The `filled=True` parameter adds semantic coloring to your tree diagram:

1. **Hue Represents Class:**
   * Nodes dominated by Class 0 (Fail) are colored in one hue (e.g., orange).
   * Nodes dominated by Class 1 (Pass) are colored in another hue (e.g., blue).
2. **Saturation Represents Purity:**
   * A node with high impurity ($\text{Gini} \approx 0.5$) appears light or washed out.
   * A completely pure leaf node ($\text{Gini} = 0.0$) has deep, rich color saturation.

> 💡 **Visual Insight:** You can quickly scan a large tree and identify pure leaves by looking for the darkest, most saturated colored boxes!

---

## 6️⃣ Complete Working Code: Train & Visualize

Here is a complete, runnable script that creates a dataset, trains a Decision Tree, and plots the graph:

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Dataset: Study Hours vs Exam Result (0 = Fail, 1 = Pass)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1, 1, 1]

# 2. Instantiate and Train Model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# 3. Predict on a test sample
test_hours = [[3.5]]
prediction = model.predict(test_hours)
print(f"Prediction for {test_hours[0][0]} hours: {prediction[0]} ({'Pass' if prediction[0] == 1 else 'Fail'})")

# 4. Plot the Decision Tree
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

## 7️⃣ Alternative: Text Representation with `export_text`

If you are working in an environment without a graphical display (like a terminal or headless cloud server), you can view the tree as plain text using `export_text`:

```python
from sklearn.tree import export_text

tree_rules = export_text(model, feature_names=["Study Hours"])
print(tree_rules)
```

### Output:
```text
|--- Study Hours <= 3.50
|   |--- class: 0
|--- Study Hours >  3.50
|   |--- class: 1
```

---

## 🎯 Core Memory Points

1. Decision Trees are **interpretable white-box models**, making them easy to debug, validate, and explain.
2. Standard classification metrics (**Accuracy**, **Confusion Matrix**, **Precision**, **Recall**, **F1**) apply directly to Decision Trees.
3. `sklearn.tree.plot_tree` visualizes the complete decision tree along with impurity, sample distributions, and class labels.
4. Each node in `plot_tree` displays: split condition, `gini`, `samples`, class `value` breakdown, and dominant `class`.
5. `filled=True` colors nodes by class, with darker saturation indicating higher purity ($\text{Gini} \to 0$).

---

## 💼 Top Interview Questions & Answers

### Q1: Why is interpretability an advantage of Decision Trees?
> **Answer:** Decision Trees provide transparent, sequential rules that can be audited step-by-step, making it easy to explain to non-technical stakeholders why a particular prediction was made.

### Q2: How do you visualize a Decision Tree in scikit-learn?
> **Answer:** Using `sklearn.tree.plot_tree()` along with `matplotlib.pyplot` for visual diagram rendering, or `sklearn.tree.export_text()` for terminal text output.

### Q3: What information is contained in each node box of `plot_tree`?
> **Answer:** The decision rule/condition, the Gini impurity score, the number of training samples in that node, the class sample distribution vector (`value`), and the majority class assignment.

### Q4: What does the color intensity mean when `filled=True` is enabled in `plot_tree`?
> **Answer:** Color hue indicates the majority class, while color saturation represents node purity: deeper, more intense shades indicate lower Gini impurity (higher purity).

### Q5: Can Decision Trees be used for regression as well as classification?
> **Answer:** Yes. `scikit-learn` provides `DecisionTreeRegressor`, which uses variance reduction or Mean Squared Error (MSE) instead of Gini impurity to evaluate splits.

---

## 🗺️ Day 4 Learning Roadmap

```text
Day 4: Decision Trees
├── Part 1: Decision Tree Basics & Intuition    ✅ (Completed)
├── Part 2: How Decision Trees Learn (Splits)   ✅ (Completed)
├── Part 3: Gini Impurity Manual Calculation    ✅ (Completed)
├── Part 4: Max Depth & Overfitting             ✅ (Completed)
├── Part 5: Model Evaluation & Visualization    ✅ (Completed)
└── Part 6: End-to-End Mini Project             ⏳ (Next)
```

👉 **Next Up:** [Part 6: End-to-End Decision Tree Mini-Project](file:///d:/Vignesh%20repositories/AI%20-%20Learning/Day%204/Part%206.md)