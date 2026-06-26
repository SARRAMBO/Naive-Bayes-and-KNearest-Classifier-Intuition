
# Naive Bayes & K-Nearest Neighbors (KNN) Implementation

## 📌 Overview

This repository contains a comprehensive Jupyter Notebook exploring two foundational machine learning algorithms: **Naive Bayes (Probabilistic)** and **K-Nearest Neighbors (Distance-Based)**.

The project breaks down the mathematical intuition behind the algorithms and provides practical, step-by-step Python implementations for both classification and regression tasks. It also emphasizes model optimization through exhaustive hyperparameter tuning using cross-validation.

## 🚀 Core Concepts Covered

### 1. Naive Bayes Classifier

* **Mathematical Intuition:** Explores the core engine of the algorithm (Bayes' Theorem) and the "naive" assumption of feature independence.
* **Implementation:** Utilizes `GaussianNB` from Scikit-Learn on the Iris dataset for multi-class classification.
* **Hyperparameter Tuning:** Uses `GridSearchCV` to optimize:
* `var_smoothing`: Regularization parameter to widen Gaussian curves and handle extreme outliers.
* `priors`: Adjusting class weights to handle imbalanced datasets.


* **Feature Scaling:** Explains the mathematical reasoning behind why Naive Bayes *does not* require standardization of training/test data.

### 2. K-Nearest Neighbors (KNN)

* **Classification (`KNeighborsClassifier`):** Implemented on a synthetic binary dataset.
* **Regression (`KNeighborsRegressor`):** Implemented on a synthetic continuous dataset.
* **Distance Metrics:** Comparison and implementation of **Euclidean** (best for continuous/low-dimensional data) vs. **Manhattan** (best for high-dimensional/outlier-heavy data).
* **Spatial Data Structures:** Evaluates search algorithm efficiency between **K-D Trees** (for low dimensions) and **Ball Trees** (for high dimensions).
* **Hyperparameter Tuning:** Automates the selection of `n_neighbors`, `metric`, and `algorithm` using `GridSearchCV` to maximize model accuracy and minimize error.

## 📊 Evaluation Metrics Used

The models are evaluated using standard Scikit-Learn metrics to ensure robustness:

* **Classification:** Accuracy Score, Confusion Matrix, Classification Report (Precision, Recall, F1-Score).
* **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-Squared ($R^2$) Score.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

## ⚙️ How to Run

1. Clone this repository to your local machine.
2. Ensure you have Python 3.x installed along with the required libraries. You can install the dependencies using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

```


3. Launch Jupyter Notebook:
```bash
jupyter notebook

```


4. Open `Naive_Bayes_and_K_Nearest_Neighbours_Machine_Learning.ipynb` and run the cells sequentially.
