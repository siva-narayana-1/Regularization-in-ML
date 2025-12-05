# Regularization-in-ML
---

# 📘 Linear Regression From Scratch with L1 & L2 Regularization

*A complete implementation of Gradient Descent, Lasso (L1), and Ridge (L2) using NumPy.*

This project demonstrates how linear regression works **internally**, without using machine learning libraries.
Everything is implemented from scratch:

* Manual predictions
* Manual gradient computation
* Manual L1 and L2 regularization
* Weight update rules
* Visualization of weight shrinkage

This helps build a strong foundational understanding of how ML models learn.

---

# 📊 Dataset — House Price Prediction

We use a simple dataset with 3 features:

| Size (sqft) | Bedrooms | Age (years) | Price  |
| ----------- | -------- | ----------- | ------ |
| 800         | 1        | 10          | 120000 |
| 1000        | 2        | 15          | 150000 |
| 1200        | 2        | 20          | 180000 |
| 1500        | 3        | 18          | 250000 |
| 1800        | 3        | 30          | 300000 |
| 2000        | 4        | 8           | 350000 |

### Feature matrix `X`:

```python
X = np.array([
    [800, 1, 10],
    [1000, 2, 15],
    [1200, 2, 20],
    [1500, 3, 18],
    [1800, 3, 30],
    [2000, 4, 8]
], float)
```

### Target vector `y`:

```python
y = np.array([120000,150000,180000,250000,300000,350000], float)
```

---

# 🧠 Mathematical Foundation

### **Model (Hypothesis)**

[
\hat{y} = Xw + b
]

* `w` → weight vector (3 weights)
* `b` → bias (scalar)

---

### **Loss Function — Mean Squared Error (MSE)**

[
L = \frac{1}{n}\sum (y - \hat{y})^2
]

---

### **Gradient of MSE**

[
\frac{\partial L}{\partial w} = -\frac{2}{n}X^T(y-\hat{y})
]

[
\frac{\partial L}{\partial b} = -\frac{2}{n}\sum (y-\hat{y})
]

These gradients are used for updating weights via gradient descent.

---

# 🔧 Regularization (L1 & L2)

Regularization reduces overfitting by penalizing large weights.

## 🟦 L2 Regularization (Ridge)

Penalty:

[
\lambda \sum w^2
]

Gradient:

[
dw_{L2} = dw + 2\lambda w
]

Effect:

* Smoothly shrinks weights
* Makes model stable
* Handles multicollinearity

---

## 🟧 L1 Regularization (Lasso)

Penalty:

[
\lambda \sum |w|
]

Gradient:

[
dw_{L1} = dw + \lambda \cdot sign(w)
]

Effect:

* Forces weights to zero
* Performs feature selection
* Produces sparse models

---

# 🧪 Implementation — Gradient Descent with Regularization

### L2 Regularization Example:

```python
import numpy as np

n, d = X.shape
w = np.zeros(d)
b = 0.0

lr = 1e-7
lam = 0.01

for epoch in range(7000):

    y_pred = X @ w + b

    dw = (-2/n) * (X.T @ (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # L2 penalty
    dw_l2 = dw + 2 * lam * w

    if np.linalg.norm(dw_l2) < 1e-6:
        break

    w -= lr * dw_l2
    b -= lr * db

print("Final w:", w)
print("Final b:", b)
```

---

# 📉 Visualizing Weight Shrinkage

```python
import matplotlib.pyplot as plt

w_history = []

for epoch in range(7000):

    y_pred = X @ w + b

    dw = (-2/n) * (X.T @ (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    dw_l2 = dw + 2 * lam * w

    w -= lr * dw_l2
    b -= lr * db

    w_history.append(w.copy())

w_history = np.array(w_history)

plt.plot(w_history[:,0], label="Weight 1 (Size)")
plt.plot(w_history[:,1], label="Weight 2 (Bedrooms)")
plt.plot(w_history[:,2], label="Weight 3 (Age)")

plt.xlabel("Epochs")
plt.ylabel("Weight Value")
plt.title("L2 Regularization: Weight Shrinkage")
plt.legend()
plt.show()
```

---

# 🎯 Making Predictions

```python
def predict(X_new, w, b):
    return X_new @ w + b
```

Example:

```python
new_house = np.array([1600, 3, 12])
price = predict(new_house, w, b)

print("Predicted Price:", price)
```

---

# 🔬 L1 vs L2 — Quick Comparison

| Feature           | L1 (Lasso)            | L2 (Ridge)    |   |       |
| ----------------- | --------------------- | ------------- | - | ----- |
| Penalty           | (                     | w             | ) | (w^2) |
| Weight Shrinkage  | Strong                | Smooth        |   |       |
| Zero Weights      | ✔ Yes                 | ❌ No          |   |       |
| Feature Selection | ✔ Yes                 | ❌ No          |   |       |
| Multicollinearity | ❌ Weak                | ✔ Strong      |   |       |
| Ideal Use Case    | High-dimensional data | Stable models |   |       |

---

# 🏁 Conclusion

In this project, we implemented:

* Multi-feature Linear Regression
* Gradient Descent from scratch
* L1 Regularization (Lasso)
* L2 Regularization (Ridge)
* Visualization of weight behavior
* Prediction on new data

This demonstrates how machine learning models work at a mathematical and computational level — without relying on libraries like scikit-learn.

---

# 📬 Contact
**Done By: M.Siva Narayana Surya Chandra**

If you find this project useful or want to collaborate:

**📧 Email:** [sivanarayanam27@gmail.com](mailto:sivanarayanam27@gmail.com)


