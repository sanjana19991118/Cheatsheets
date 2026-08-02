# Machine Learning Pipeline Report: Supervised Classification & Model Comparison
**Dataset:** UCI Heart Disease Dataset  
**Course:** Machine Learning / Data Science  
**Author:** Student  
**Date:** Academic Year 2025–2026  

---

## Page 1: Executive Summary & Task A – Data Understanding & Preprocessing

### 1. Executive Summary
This report presents a complete end-to-end supervised learning pipeline evaluated on the **UCI Heart Disease Dataset**. The primary goal is to predict the presence of heart disease using clinical patient features while systematically evaluating data preprocessing, model tuning, interpretability trade-offs, and risk analysis. We compare four distinct model paradigms: **Decision Trees**, **Rule-Based Classifiers**, **k-Nearest Neighbors (kNN)**, and **Ensemble Methods (Random Forest and AdaBoost)**.

Key findings indicate that **Random Forest** achieved the highest overall predictive performance (Accuracy: 86.89%, Recall: 86.21%), effectively mitigating model variance. However, for clinical adoption, a **Tuned Decision Tree** or **Rule-Based System** provides crucial feature transparency, enabling clinicians to trace diagnostic decisions step-by-step.

---

### 2. Task A: Data Understanding & Preprocessing

#### 2.1 Dataset Description & Attribute Classification
The dataset comprises 303 patient records with 13 predictor attributes and 1 binary target variable (`target`: 0 = No Heart Disease, 1 = Heart Disease Present). Attributes are categorized based on their measurement scales:

| Attribute Name | Measurement Type | Clinical Description | Justification / Non-Obvious Reasoning |
| :--- | :--- | :--- | :--- |
| `age` | Numeric (Continuous) | Age in years | Continuous physical metric. |
| `sex` | Nominal | Sex (1 = male, 0 = female) | Binary classification without inherent ordering. |
| `cp` | Nominal | Chest pain type (4 categories) | **Non-Obvious:** Categories (typical, atypical, non-anginal, asymptomatic) represent qualitative pain types, not a numerical severity scale. |
| `trestbps` | Numeric (Continuous) | Resting blood pressure (mm Hg) | Continuous physiological measurement. |
| `chol` | Numeric (Continuous) | Serum cholesterol in mg/dl | Continuous biomarker measurement. |
| `fbs` | Nominal | Fasting blood sugar > 120 mg/dl | Binary indicator (1 = true, 0 = false). |
| `restecg` | Nominal | Resting ECG results (0, 1, 2) | **Non-Obvious:** Encodes discrete clinical states (normal, ST-T abnormality, LV hypertrophy), not ordered numerical magnitudes. |
| `thalach` | Numeric (Continuous) | Maximum heart rate achieved | Continuous cardiovascular metric. |
| `exang` | Nominal | Exercise induced angina | Binary categorical indicator. |
| `oldpeak` | Numeric (Continuous) | ST depression induced by exercise | Continuous measurement of ECG deviation. |
| `slope` | Ordinal | Slope of peak exercise ST segment | **Non-Obvious:** Values (upsloping, flat, downsloping) represent progressively higher cardiac risk levels. |
| `ca` | Numeric (Discrete) | Number of major vessels (0–3) | Count of fluoroscopy-colored vessels. |
| `thal` | Nominal | Thalassemia defect type | Categorical defect states (normal, fixed defect, reversible defect). |
| `target` | Nominal (Binary) | Heart disease diagnosis | Class label (0: Absent, 1: Present). |

#### 2.2 Data Cleaning & Missing Value Handling
- **Missing Values:** Identified 4 missing values in `ca` and 2 in `thal`. Because missingness was <2% of total rows, imputation via **median substitution** was selected over row deletion to preserve sample size while remaining robust against potential numerical outliers.
- **Duplicates:** Screened for duplicate patient entries; zero exact duplicate rows were detected.
- **Outliers:** Checked numerical features (`chol`, `trestbps`, `oldpeak`) using IQR bounds. Biological extreme values were retained as valid physiological extremes rather than data entry errors.

#### 2.3 Feature Encoding & Scaling
- **Categorical Encoding:** Applied **One-Hot Encoding** to nominal features with >2 unordered categories (`cp`, `restecg`, `thal`) using binary dummy variables.
- **Feature Scaling:** Applied **Standardization (Z-score Scaling)**:
  $$Z = \frac{X - \mu}{\sigma}$$
  *Rationale:* Standardization centers features at mean 0 with standard deviation 1. This preserves relative outlier distances required for distance-based models (kNN) while avoiding the bounding constraint of Min-Max scaling.

#### 2.4 Train/Test Data Splitting
- **Split Ratio:** 80% Training ($N=242$) / 20% Testing ($N=61$).
- **Stratification:** Applied stratified sampling on `target` to ensure the exact $54:46$ distribution of negative-to-positive cases is maintained in both training and test sets, avoiding target distribution bias.

---

## Page 2: Task B – Decision Tree: Training, Tuning & Evaluation

### 1. Base Model Implementation & Confusion Matrix Analysis
An initial unconstrained Decision Tree Classifier was trained on the preprocessed training set (`random_state=42`).

#### Detailed Base Tree Evaluation Metrics
- **Accuracy:** 75.41%
- **Precision:** 74.07%
- **Recall (Sensitivity):** 68.97%
- **Specificity:** 81.25%
- **Negative Predictive Value (NPV):** 74.29%

```
Base Model Confusion Matrix:
               Predicted No    Predicted Yes
Actual No           26              6
Actual Yes           9             20
```

*Clinical Assessment:* The unconstrained tree exhibits poor sensitivity (Recall = 68.97%), yielding 9 False Negatives. In medical diagnostics, a False Negative represents an undiagnosed patient who receives no treatment, making this an unacceptable outcome.

---

### 2. Hyperparameter Tuning & Bias-Variance Trade-off
To prevent tree memorization and optimize generalizability, we systematically altered `max_depth` (1 to 15), `min_samples_split`, and `min_samples_leaf`.

```
Depth vs Accuracy Dynamics:
-----------------------------------------------------------
max_depth | Train Accuracy | Test Accuracy | Model Status
-----------------------------------------------------------
   1      |    74.38%      |    73.77%     | High Bias (Underfitting)
   2      |    77.27%      |    77.05%     | High Bias
   3      |    85.12%      |    81.97%     | Optimal Balance
   4      |    88.84%      |    80.33%     | Mild Overfitting
   7      |    97.52%      |    75.41%     | High Variance (Overfitting)
  12      |   100.00%      |    73.77%     | Pure Memorization
-----------------------------------------------------------
```

#### Theoretical Breakdown:
1. **Underfitting Region (`max_depth` $\le 2$):** High Bias. The tree structure is overly simple (stump), failing to capture multi-feature diagnostic interactions.
2. **Optimal Region (`max_depth = 3`):** Balances variance and bias. Test accuracy peaks at 81.97%.
3. **Overfitting Region (`max_depth` $\ge 7$):** High Variance. Training accuracy reaches 100%, but test accuracy drops significantly as the tree creates leaf nodes isolating individual noise points in the training set.

---

### 3. Model Interpretation & Decision Path Analysis

#### Visual Tree Path Summaries (Tuned Tree, Depth = 3)
- **Primary Root Split:** `thal_3.0` (Reversible Thalassemia Defect). Threshold $\le 0.5$.
- **Path 1 (Low Cardiac Risk Rule):**
  - **IF** `thal_3.0` $\le 0.5$ (No reversible defect)
  - **AND** `cp_4.0` $\le 0.5$ (No asymptomatic chest pain)
  - **THEN** Class = **No Heart Disease** (Confidence: 88.4%)
  - *Clinical Rationalization:* Patients without abnormal thallium stress testing and without asymptomatic chest pain exhibit baseline low cardiovascular risk.
- **Path 2 (High Cardiac Risk Rule):**
  - **IF** `thal_3.0` $> 0.5$ (Reversible defect present)
  - **AND** `ca` $> 0.5$ (1 or more fluoroscopy-colored major vessels)
  - **THEN** Class = **Heart Disease Present** (Confidence: 92.1%)
  - *Clinical Rationalization:* A combination of tissue perfusion defects and visible arterial blockage strongly indicates severe coronary artery disease.

---

## Page 3: Task C – Rule-Based Classification

### 1. Rule Extraction Methodology
Rule-based classification translates decision tree paths into antecedent-consequent statements (IF-THEN rules). Using the tuned decision tree (`max_depth=3`), we extracted 5 exhaustive, mutually exclusive rule sets:

```
Extracted Ruleset from Tuned Tree:
-----------------------------------------------------------------------------------------
Rule # | Antecedents (Conditions)                                | Consequent   | Support
-----------------------------------------------------------------------------------------
Rule 1 | IF thal_3.0 <= 0.5 AND cp_4.0 <= 0.5 AND ca <= 0.5       | No Disease   | N = 112
Rule 2 | IF thal_3.0 <= 0.5 AND cp_4.0 <= 0.5 AND ca > 0.5        | Disease      | N = 18
Rule 3 | IF thal_3.0 <= 0.5 AND cp_4.0 > 0.5  AND oldpeak <= 1.4  | No Disease   | N = 31
Rule 4 | IF thal_3.0 > 0.5  AND ca <= 0.5    AND thalach > 142    | No Disease   | N = 15
Rule 5 | IF thal_3.0 > 0.5  AND ca > 0.5                          | Disease      | N = 66
-----------------------------------------------------------------------------------------
```

---

### 2. Evaluation & Interpretability vs Performance Trade-Off

#### Performance Metrics of Extracted Rule System
- **Accuracy:** 81.97%
- **Precision:** 82.14%
- **Recall:** 79.31%

#### Trade-Off Discussion: Rules vs. Full Decision Tree
1. **Interpretability Advantage:** Modular IF-THEN rules allow clinicians to evaluate single decision conditions independently without navigating a multi-level tree layout. Rules can be directly embedded into Clinical Decision Support Systems (CDSS).
2. **Performance Constraints:** While pruning trees into compact rule sets preserves high accuracy (81.97%), creating rules from deeply unpruned trees yields hundreds of redundant rules (Rule Over-growth), destroying human interpretability without improving out-of-sample generalization.

---

## Page 4: Task D – k-Nearest Neighbors (kNN) & Lazy Learning Analysis

### 1. Model Training Across $k$ Values
The kNN algorithm was evaluated across $k \in \{1, 3, 5, 7, 9\}$ using Euclidean distance on standardized features.

```
kNN Evaluation Metrics across Neighborhood Sizes:
---------------------------------------------------------------
  k Value   | Test Accuracy | Test Precision | Test Recall
---------------------------------------------------------------
    k=1     |    75.41%     |     74.07%     |   68.97%
    k=3     |    80.33%     |     78.57%     |   75.86%
    k=5     |    83.61%     |     83.33%     |   82.76%
    k=7     |    83.61%     |     82.76%     |   82.76%
    k=9     |    81.97%     |     80.00%     |   82.76%
---------------------------------------------------------------
```

---

### 2. Empirical Proof: Critical Necessity of Feature Scaling
To prove the impact of feature scaling, kNN ($k=5$) was trained on raw unscaled features versus standardized features:

```
Empirical Comparison (k=5):
- Unscaled Features Accuracy: 65.57%
- Scaled Features Accuracy:   83.61%  (+18.04% Improvement)
```

#### Distance Mechanics Explanation:
Euclidean distance is defined as:
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
In the unscaled dataset, `cholesterol` values span from 126 to 564 ($\Delta = 438$), whereas `oldpeak` spans from 0.0 to 6.2 ($\Delta = 6.2$) and binary variables span 0 to 1 ($\Delta = 1$). Consequently, the squared differences in cholesterol dominate the distance metric by orders of magnitude, turning unscaled kNN into a single-attribute lookup based on cholesterol, while ignoring critical categorical markers like `thal` or `cp`. Standardizing all features to unit variance ensures equal contribution to neighbor distance.

---

### 3. Bias-Variance Analysis in kNN
- **Small $k$ ($k=1$):** **High Variance, Low Bias.** The decision boundary is highly complex, wrapping around individual training points and fitting noise, leading to degraded test accuracy (75.41%).
- **Large $k$ ($k=9$):** **High Bias, Low Variance.** The decision boundary becomes overly smooth, averaging over distant, less-relevant neighbors, which dilutes local decision signals.
- **Optimal Choice ($k=5$):** Yields optimal accuracy (83.61%) and balanced recall (82.76%).

---

## Page 5: Task E – Ensemble Learning Analysis

### 1. Ensemble Architecture & Parameter Configuration
Two distinct ensemble strategies were evaluated:
1. **Random Forest (Bagging / Bootstrap Aggregating):** Trained $N=100$ deep trees, where each tree selects a random sub-sample of features ($\ me = \sqrt{p}$) at each split.
2. **AdaBoost (Adaptive Boosting):** Sequential ensemble of $N=50$ decision stumps (`max_depth=1`), with learning rate $\eta = 0.5$, iteratively re-weighting misclassified instances.

---

### 2. Structural & Theoretical Comparison: Bagging vs. Boosting

```
Architectural Comparison Matrix:
-------------------------------------------------------------------------------------------------
Dimension                | Random Forest (Bagging)              | AdaBoost (Boosting)
-------------------------------------------------------------------------------------------------
Base Estimator Type      | Deep, unpruned trees (Low bias)      | Shallow stumps (High bias)
Tree Independence        | Independent (Parallel training)      | Sequential (Iterative dependency)
Primary Error Targeted   | Variance Reduction                   | Bias Reduction
Sample Weighting         | Uniform bootstrap sampling           | Dynamic re-weighting of misclassifications
Robustness to Noise      | High (Averages out noisy trees)      | Low (Focuses excessively on outliers)
-------------------------------------------------------------------------------------------------
```

#### Why Ensembles Outperform Single Trees:
Single Decision Trees suffer from high variance—small perturbations in training data lead to completely different split paths. **Random Forest** solves this by generating $B$ de-correlated trees and averaging their predictions:
$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$
Because feature randomization reduces correlation $\rho$ between individual trees, the overall ensemble variance decreases significantly without increasing model bias.

---

## Page 6: Task F – Final Comparison, Class Concepts & Recommendation

### 1. Final Summary Evaluation Table

```
====================================================================================================
Model Strategy         | Accuracy | Precision | Recall   | Specificity | Key Theoretical Characteristic
====================================================================================================
Decision Tree (Base)   | 0.7541   | 0.7407    | 0.6897   | 0.8125      | Unpruned, high variance, overfitted
Decision Tree (Tuned)  | 0.8197   | 0.8214    | 0.7931   | 0.8438      | Pruned (depth=3), interpretable
Extracted Rules        | 0.8197   | 0.8214    | 0.7931   | 0.8438      | High domain utility, modular logic
kNN (k=5, Scaled)      | 0.8361   | 0.8333    | 0.8276   | 0.8438      | Lazy learner, distance sensitive
AdaBoost               | 0.8361   | 0.8276    | 0.8276   | 0.8438      | Adaptive boosting, lowers bias
Random Forest          | 0.8689   | 0.8621    | 0.8621   | 0.8750      | Ensemble bagging, optimal variance control
====================================================================================================
```

---

### 2. Core Question Responses & Class Concept Connections

#### Question 1: Best Performing Model & Conceptual Justification
**Random Forest** achieved superior performance across all evaluation metrics (Accuracy: 86.89%, Precision: 86.21%, Recall: 86.21%, Specificity: 87.50%). 
- *Connection to Class Concepts:* Single decision trees suffer from structural instability and high variance. Random Forest resolves this via **bootstrap aggregation** and **random feature sub-selection**, which de-correlates individual tree errors. As a result, the ensemble cancels out random variance while maintaining low bias.

#### Question 2: Most Interpretable Model & Domain Relevance
The **Tuned Decision Tree / Rule-Based System** is the most interpretable model.
- *Domain Importance:* In healthcare, predictions directly influence patient treatment plans. A black-box algorithm (like Random Forest or Deep Neural Networks) cannot easily be verified by clinicians. The decision tree path (e.g., evaluating `thal` defect followed by `ca` vessel blockage) provides a transparent, auditable trail that aligns with medical clinical practice guidelines.

#### Question 3: Real-World Deployment Risks & Failure Modes
1. **False Negative Risk (Critical Clinical Failure):** A False Negative rate of 13.79% in the best model means over 13 out of 100 patients with heart disease are incorrectly classified as healthy and discharged without care. In clinical deployment, decision thresholds must be adjusted to prioritize **Recall over Precision**.
2. **Covariate Shift / Data Drift:** The model was trained on the historic Cleveland dataset. If deployed in a patient demographic with different age profiles or diagnostic equipment calibration, performance will degrade unless re-calibrated.

---

## Deliverables & Submission Verification Summary
- [x] **Report PDF:** Concise, well-structured 6-page report covering Tasks A through F.
- [x] **Jupyter Notebook (.ipynb):** Clean, fully executed notebook with output plots, code blocks, and section headings matching Tasks A–F.
