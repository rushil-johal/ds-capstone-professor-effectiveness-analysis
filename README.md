# Assessing Professor Effectiveness using RateMyProfessor Data 📊

This repository contains the complete capstone project I completed as part of one of many NYU undergraduate courses (DS-UA) offered by the NYU Center for Data Science. The project leverages real-world scraped data from RateMyProfessor.com to analyze, model, and draw statistically robust conclusions about professor evaluations across the United States and even internationally. 

---

## Project Overview

Academic institutions often rely on subjective evaluations to assess teaching effectiveness. This project investigates potential **biases and predictors** of professor ratings using public RateMyProfessor (RMP) data. Through statistical testing, regression modeling, and machine learning, the project explores whether factors such as gender, teaching modality, difficulty, or "hotness" (pepper icon) influence ratings.

The project answers **10 core research questions** along with an **extra credit exploration**, combining:
- Exploratory Data Analysis
- Hypothesis Testing
- Correlation Analysis
- Linear and Logistic Regression
- Classification Modeling
- ROC/AUC Evaluation
- Feature Engineering and Preprocessing

---

## Repository Contents

| File/Folder | Description |
|-------------|-------------|
| `RJ_Capstone_Script.py` | Main Python script with all data cleaning, EDA, statistical tests, regression, classification models, and visualizations. Random number generator seeded with my NYU N-number for reproducibility. |
| `Principles of Data Science Capstone Report.pdf` | Full write-up of the analysis, including interpretation of results, visuals, and statistical reasoning across all questions. |
| `rmpCapstoneNum.csv` | Raw quantitative dataset scraped from RateMyProfessor, including average ratings, difficulty, gender, and more (n = ~89,000 professors). |
| `rmpCapstoneQual.csv` | Raw qualitative dataset containing field of study, university, and state information for the same professors. |
| `cleaned_rmp_data.csv` | Final cleaned and preprocessed dataset used in all analyses. Missing values dropped, thresholds applied (e.g. minimum 5 ratings), and boolean fields encoded. |
| `*.png` | Auto-generated visualizations used in the report (e.g., boxplots, scatterplots, ROC curves, regression fits). |

---

## Key Technologies

- **Python (pandas, numpy, matplotlib, seaborn)**
- **scikit-learn** for regression and classification
- **statsmodels** for OLS analysis
- **Statistical testing**: t-tests, Spearman correlation
- **Data cleaning and merging**: CSV preprocessing and transformation

---

## Sample Questions Answered

1. **Is there evidence of gender bias in professor ratings?**
2. Does teaching experience (proxy: number of ratings) relate to quality?
3. Do more "difficult" professors receive worse ratings?
4. Are professors teaching online rated differently?
5. Can we predict whether students would retake a professor based on their ratings?
6. How does being marked “hot” influence evaluations?
7. How predictive is difficulty alone in determining a professor's rating?
8. Can we improve predictions using *all* available features?
9. Can we accurately classify who receives a “pepper” using just average rating?
10. Do multiple factors improve that classification?

---

## Highlights & Key Findings

- **Statistically significant but small pro-male bias** was observed in ratings.
- **High difficulty correlates with lower ratings** (ρ ≈ -0.61).
- **Proportion of students willing to retake a class** is the most predictive variable (β ≈ 0.616).
- Logistic regression models showed **AUROC up to 0.78**, predicting professor "hotness" with reasonable accuracy.
- **Full regression model (R² = 0.81)** outperformed simpler ones, showcasing your understanding of multivariable modeling and feature collinearity.

---

## Extra Insight

As part of the extra credit, I analyzed **geographic variation** in ratings across U.S. states. Surprisingly, some UK-affiliated locations were present (e.g., Derbyshire and Edinburgh), highlighting anomalies and opportunities for deeper institutional-level filtering or cultural analysis in evaluation norms.

---
