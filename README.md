
# Crime Arrest Prediction Using Machine Learning

## 1.0 Overview
This project evaluates and compares the performance of three supervised machine learning models — **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Decision Tree** — to predict whether a crime will result in an arrest based on historical crime data.

The dataset (`crimes.csv`) contains over **6 million rows** of crime-related data from Chicago. Each model was tested across different combinations of:
- **Train-test split ratios**: 60/40, 70/30, and 80/20
- **Top N features**: 6, 8, and 10
- **Model-specific hyperparameters** (K for KNN, C for SVM, and depth for Decision Tree)

The goal was to identify the best model configuration using key evaluation metrics: **Accuracy, Weighted Precision, Weighted Recall, and Weighted F1 Score**.

## 2.0 Champion Model Summary
| Metric              | KNN (Best) | SVM (Best) | Decision Tree (Best) |
|---------------------|------------|------------|-----------------------|
| Train/Test Split    | 70/30      | 60/40      | 60/40                 |
| Features Used (N)   | 8          | 6          | 10                    |
| Accuracy            | 0.8799     | 0.7627     | **0.8810**            |
| Weighted Precision  | 0.8788     | 0.7707     | **0.8822**            |
| Weighted Recall     | **0.8799** | 0.7621     | 0.8811                |
| Weighted F1 Score   | **0.8751** | 0.7657     | 0.8749                |
| Best Hyperparameter | K = 9      | C = 0.1    | Depth = 10            |

**Champion Model**: Decision Tree (Depth=10, 60/40 split, N=10)

## 3.0 Confusion Matrix Analysis
### KNN (K=9, N=8, 70/30 split)
- True Negatives: 1,332,195
- False Positives: 57,321
- False Negatives: 174,764
- True Positives: 368,801

### SVM (C=0.1, N=6, 60/40 split)
- True Negatives: 1,506,670
- False Positives: 346,447
- False Negatives: 266,803
- True Positives: 457,521

### Decision Tree (Depth=10, N=10, 60/40 split)
- True Negatives: 1,795,712
- False Positives: 56,976
- False Negatives: 249,589
- True Positives: 475,164

## 4.0 Model Performance Discussion
### 4.1 Accuracy Comparison
- Decision Tree had the highest accuracy: **88.10%**
- KNN: 87.99%
- SVM: 76.21%

### 4.2 Weighted Precision
- Decision Tree: **88.22%** (highest)
- KNN: 87.88%
- SVM: 77.07%

### 4.3 Weighted Recall
- KNN: **87.99%** (highest)
- Decision Tree: 88.11%
- SVM: 76.21%

### 4.4 Weighted F1 Score
- KNN: **87.51%** (highest)
- Decision Tree: 87.49%
- SVM: 76.57%

## 5.0 Key Insights
1. KNN performs best when trained with more data (70/30 split), due to its instance-based nature.
2. SVM underperformed due to higher false positives and false negatives.
3. Decision Tree had the best balance across all metrics.

## 6.0 Recommendations
- Automate hyperparameter tuning using Grid Search or Bayesian Optimization.
- Improve feature selection using model-based methods like Random Forest or Gradient Boosted Trees.
- Explore more feature combinations beyond the top 10.


## 7.0 Conclusion
Decision Tree emerged as the most reliable model with **88.10% accuracy**. The experiment highlighted how data splits, feature selection, and parameter tuning influence model effectiveness.

## 8.0 References (IEEE Style)
1. M. Janssen et al., "Benefits, Adoption Barriers and Myths of Open Data," *Gov. Info Q.*, vol. 29, no. 4, 2012. [Online]. Available: https://doi.org/10.1016/j.giq.2018.01.004
2. GeeksforGeeks. (2023). *K-Nearest Neighbors (KNN) Algorithm*. [Online]. Available: https://www.geeksforgeeks.org/k-nearest-neighbours/
3. Scikit-learn documentation. *StandardScaler*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
4. Forecastegy. (2023). *Why Feature Scaling Matters*. [Online]. Available: https://forecastegy.com/why-feature-scaling-matters
5. Creative Commons. (n.d.). *CC0 1.0 Universal*. [Online]. Available: https://creativecommons.org/publicdomain/zero/1.0/
6. Z. Martin. (n.d.). *When to Use the Chi-Square Test*. Statology. [Online]. Available: https://www.statology.org/when-to-use-chi-square-test/
7. Vitalflux. (n.d.). *Micro vs. Macro Averaging in Multi-Class Classification*. [Online]. Available: https://vitalflux.com/micro-average-macro-average-scoring-metrics-multi-class-classification-python/
8. J. Brownlee. (2020). *Confusion Matrix for Classification*. [Online]. Available: https://machinelearningmastery.com/confusion-matrix-for-machine-learning/
9. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *J. Mach. Learn. Res.*, vol. 12, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html
10. R.-E. Fan et al., “LIBLINEAR: A Library for Large Linear Classification,” *J. Mach. Learn. Res.*, vol. 9, 2008. [Online]. Available: https://jmlr.csail.mit.edu/papers/v9/fan08a.html
