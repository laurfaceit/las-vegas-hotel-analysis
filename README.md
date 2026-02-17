Las Vegas Hotel Review Analysis

Project Overview

This project analyzes TripAdvisor hotel reviews from Las Vegas in order to identify the factors that influence hotel ratings (Score).

The analysis includes:
- Data cleaning and preprocessing
- Descriptive statistical analysis
- Correlation analysis
- Multiple Linear Regression (OLS)
- Model diagnostics
- Customer segmentation using K-Means clustering

---

Dataset

- 504 observations
- Dependent variable: **Score**
- Independent variables: numerical and categorical variables related to tourists and reviews

---

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn

---

Regression Analysis

A Multiple Linear Regression model (OLS) was estimated to evaluate the impact of independent variables on hotel ratings.

Key findings:
- Statistically significant variables were selected based on p-value < 0.05
- No autocorrelation (Durbin-Watson ≈ 2)
- No heteroscedasticity (Breusch-Pagan test)
- No severe multicollinearity (VIF < 5)

---

Clustering Analysis

K-Means clustering identified 3 distinct tourist segments:

1. Highly active and critical reviewers
2. Low-activity tourists with variable ratings
3. Medium-activity tourists with generally higher satisfaction

---

Business Insights

- Highly active users tend to provide more critical evaluations.
- Hotel managers should focus on improving experience consistency.
- Customer segmentation can support targeted marketing strategies.

---

1. Install required libraries:
