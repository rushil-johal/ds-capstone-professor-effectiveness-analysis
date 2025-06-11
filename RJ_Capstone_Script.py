# CAPSTONE PROJECT - CODING SCRIPTS
# RUSHIL JOHAL
# NYU CENTER FOR DATA SCIENCE
# DECEMBER 2024



# IMPORTED PACKAGES
import pandas as pd
import numpy as np
import random

from scipy.stats import ttest_ind, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS, add_constant

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, mean_squared_error, r2_score



# DATA PREPROCESSING AND CLEANING

# Seed the random number generator with my N-number
np.random.seed(19128261)

# Dataset File paths
num_data_path = 'rmpCapstoneNum.csv'
qual_data_path = 'rmpCapstoneQual.csv'

# Load the datasets
num_data = pd.read_csv(num_data_path)
qual_data = pd.read_csv(qual_data_path)

# Rename columns for clarity going forward based on the dataset description
num_data.columns = [
    "Average_Rating",         # Average Rating
    "Average_Difficulty",     # Average Difficulty
    "Number_of_Ratings",      # Number of Ratings
    "Received_Pepper",        # Received a "Pepper"?
    "Proportion_Take_Again",  # Proportion of students who would take class again
    "Online_Ratings",         # Number of ratings from online classes
    "Male",                   # Male gender
    "Female"                  # Female gender
]

qual_data.columns = [
    "Major_Field",  # Major/Field
    "University",   # University
    "State"         # US State (2-letter abbreviation)
]

# Drop rows with missing critical data in numerical dataset
num_data_cleaned = num_data.dropna(subset=["Average_Rating", "Number_of_Ratings"])

# Filter out rows where Number_of_Ratings is less than a chosen threshold (e.g. 5 ratings) to ensure reliable average ratings
num_data_cleaned = num_data_cleaned[num_data_cleaned["Number_of_Ratings"] > 5]

# Merge numerical and qualitative datasets on their indices
merged_data = pd.merge(num_data_cleaned, qual_data, left_index=True, right_index=True)

# Ensure Boolean fields are properly formatted
merged_data["Received_Pepper"] = merged_data["Received_Pepper"].fillna(0).astype(int)
merged_data["Male"] = merged_data["Male"].fillna(0).astype(int)
merged_data["Female"] = merged_data["Female"].fillna(0).astype(int)

# Save the cleaned data to a new CSV to use going forward
merged_data.to_csv("cleaned_rmp_data.csv", index=False)

# Print summary of final cleaned data
print(merged_data.info())
print(merged_data.head())

# Histogram to visualize the Distribution of Average_Rating
plt.hist(merged_data["Average_Rating"], bins=20, edgecolor='black')
plt.title("Distribution of Average Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.savefig("average_rating_distribution.png")
plt.show()



# QUESTION 1)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Separate ratings by gender
male_ratings = cleaned_data[cleaned_data["Male"] == 1]["Average_Rating"]
female_ratings = cleaned_data[cleaned_data["Female"] == 1]["Average_Rating"]

# Perform a two-sample t-test
t_stat, p_value = ttest_ind(male_ratings, female_ratings, equal_var=False)

# Summary statistics
male_mean = male_ratings.mean()
female_mean = female_ratings.mean()
male_std = male_ratings.std()
female_std = female_ratings.std()

# Print Summary statistics
print(f"t-statistic: {t_stat}, p-value: {p_value}")
print(f"Male mean: {male_mean}, Female mean: {female_mean}")
print(f"Male std: {male_std}, Female std: {female_std}")

# Generate a boxplot to visualize the difference in ratings
plt.figure(figsize=(8, 6))
plt.boxplot([male_ratings.dropna(), female_ratings.dropna()], labels=["Male Professors", "Female Professors"])
plt.title("Distribution of Average Ratings by Gender")
plt.ylabel("Average Rating")
plt.grid(True)
plt.savefig("gender_bias_ratings_boxplot.png")



# QUESTION 2)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Calculate the correlation between Number_of_Ratings (proxy for experience) and Average_Rating
correlation, p_value = spearmanr(cleaned_data["Number_of_Ratings"], cleaned_data["Average_Rating"])

# Summary statistics
ratings_median = cleaned_data["Number_of_Ratings"].median()

# Print Summary statistics
print(f"Spearman correlation: {correlation}, p-value: {p_value}")
print(f"Median number of ratings: {ratings_median}")

# Plot the relationship between Number_of_Ratings and Average_Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=cleaned_data["Number_of_Ratings"],
    y=cleaned_data["Average_Rating"],
    alpha=0.5
)
plt.title("Experience (Number of Ratings) vs. Teaching Quality")
plt.xlabel("Number of Ratings (Proxy for Experience)")
plt.ylabel("Average Rating (Teaching Quality)")
plt.grid(True)
plt.savefig("experience_vs_quality.png")



# QUESTION 3)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Calculate the correlation between Average_Rating and Average_Difficulty
correlation, p_value = spearmanr(cleaned_data["Average_Rating"], cleaned_data["Average_Difficulty"])

# Summary statistics
difficulty_mean = cleaned_data["Average_Difficulty"].mean()
rating_mean = cleaned_data["Average_Rating"].mean()

# Print Summary statistics
print(f"Spearman correlation: {correlation}, p-value: {p_value}")
print(f"Mean difficulty: {difficulty_mean}, Mean rating: {rating_mean}")

# Plot the relationship between Average_Rating and Average_Difficulty
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=cleaned_data["Average_Difficulty"],
    y=cleaned_data["Average_Rating"],
    alpha=0.5
)
plt.title("Average Difficulty vs. Average Rating")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.grid(True)
plt.savefig("rating_vs_difficulty.png")



# QUESTION 4)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Use median number of online rating to split professors into "high online activity" and "low online activity"
online_threshold = cleaned_data["Online_Ratings"].median()

# Create groups based on online activity
high_online_group = cleaned_data[cleaned_data["Online_Ratings"] > online_threshold]["Average_Rating"]
low_online_group = cleaned_data[cleaned_data["Online_Ratings"] <= online_threshold]["Average_Rating"]

# Perform a two-sample t-test
t_stat, p_value = ttest_ind(high_online_group, low_online_group, equal_var=False)

# Calculate group means for additional insight
high_online_mean = high_online_group.mean()
low_online_mean = low_online_group.mean()

# Print Summary statistics
print(f"t-statistic: {t_stat}, p-value: {p_value}")
print(f"High Online Activity mean: {high_online_mean}")
print(f"Low Online Activity mean: {low_online_mean}")

# Plot the comparison of ratings between the two groups
plt.figure(figsize=(8, 6))
plt.boxplot(
    [high_online_group.dropna(), low_online_group.dropna()],
    labels=["High Online Activity", "Low Online Activity"]
)
plt.title("Distribution of Average Ratings by Online Activity Level")
plt.ylabel("Average Rating")
plt.grid(True)
plt.savefig("online_activity_vs_ratings.png")



# QUESTION 5)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Filter data to ensure valid proportions for Take Again
filtered_data = cleaned_data.dropna(subset=["Proportion_Take_Again"])

# Calculate the correlation between Average_Rating and Proportion_Take_Again
correlation, p_value = spearmanr(
    filtered_data["Average_Rating"], filtered_data["Proportion_Take_Again"]
)

# Summary statistics
take_again_mean = filtered_data["Proportion_Take_Again"].mean()
rating_mean = filtered_data["Average_Rating"].mean()

# Print Summary statistics
print(f"Spearman correlation: {correlation}, p-value: {p_value}")
print(f"Mean proportion take again: {take_again_mean}")
print(f"Mean rating: {rating_mean}")

# Plot the relationship between Average_Rating and Proportion_Take_Again
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=filtered_data["Proportion_Take_Again"],
    y=filtered_data["Average_Rating"],
    alpha=0.5
)
plt.title("Proportion that Take Again vs. Average Rating")
plt.xlabel("Proportion Take Again (%)")
plt.ylabel("Average Rating")
plt.grid(True)
plt.savefig("rating_vs_take_again.png")



# QUESTION 6)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Separate ratings based on whether professors received a "Pepper" (hot)
hot_group = cleaned_data[cleaned_data["Received_Pepper"] == 1]["Average_Rating"]
not_hot_group = cleaned_data[cleaned_data["Received_Pepper"] == 0]["Average_Rating"]

# Perform a two-sample t-test
t_stat, p_value = ttest_ind(hot_group, not_hot_group, equal_var=False)

# Summary statistics
hot_mean = hot_group.mean()
not_hot_mean = not_hot_group.mean()

# Print Summary statistics
print(f"t-statistic: {t_stat}, p-value: {p_value}")
print(f"Mean rating (hot): {hot_mean}")
print(f"Mean rating (not hot): {not_hot_mean}")

# Plot the comparison of ratings between the two groups
plt.figure(figsize=(8, 6))
plt.boxplot(
    [hot_group.dropna(), not_hot_group.dropna()],
    labels=["Hot Professors (Received Pepper)", "Not Hot Professors"]
)
plt.title("Distribution of Average Ratings by Hotness (Received Pepper)")
plt.ylabel("Average Rating")
plt.grid(True)
plt.savefig("hotness_vs_ratings.png")



# QUESTION 7)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Prepare data for regression
X = cleaned_data[["Average_Difficulty"]]
y = cleaned_data["Average_Rating"]

# Split data into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19128261)

# Build the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R2 and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Plot regression results
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label="Actual Ratings", alpha=0.5)
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.title("Linear Regression Model of Average Difficulty vs. Average Rating")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.legend()
plt.grid(True)
plt.savefig("regression_rating_vs_difficulty.png")

# Regression coefficients
slope = model.coef_[0]
intercept = model.intercept_

# Print Summary statistics
print(f"R2: {r2}, RMSE: {rmse}")
print(f"Slope: {slope}, Intercept: {intercept}")



# QUESTION 8)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Prepare data for regression with all factors
X_all = cleaned_data.drop(columns=["Average_Rating", "Major_Field", "University", "State"])
y_all = cleaned_data["Average_Rating"]

# Drop rows with missing values in the predictors
X_all_cleaned = X_all.dropna()
y_all_cleaned = y_all[X_all_cleaned.index]

# Standardize the cleaned predictors
scaler = StandardScaler()
X_all_scaled_cleaned = scaler.fit_transform(X_all_cleaned)

# Add constant for ordinary least squares (OLS) regression
X_all_scaled_cleaned_const = add_constant(X_all_scaled_cleaned)

# Fit the OLS regression model
model_all_cleaned = OLS(y_all_cleaned, X_all_scaled_cleaned_const).fit()

# Make predictions
y_pred_all_cleaned = model_all_cleaned.predict(X_all_scaled_cleaned_const)

# Calculate R2 and RMSE
r2_all_cleaned = model_all_cleaned.rsquared
rmse_all_cleaned = np.sqrt(mean_squared_error(y_all_cleaned, y_pred_all_cleaned))

# Extract coefficients
coefficients_cleaned = model_all_cleaned.params

# Summary of the model
model_summary_cleaned = model_all_cleaned.summary()

# Print Summary statistics
print(f"R2: {r2_all_cleaned}, RMSE: {rmse_all_cleaned}")
print(f"Coefficients: {coefficients_cleaned}")

# Plot actual vs. predicted ratings
plt.figure(figsize=(7, 6))
plt.scatter(y_all_cleaned, y_pred_all_cleaned, alpha=0.5)
plt.plot([min(y_all_cleaned), max(y_all_cleaned)], [min(y_all_cleaned), max(y_all_cleaned)], color="red", label="Prediction Line")
plt.title("Actual vs. Predicted Ratings (All Factors)")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.legend()
plt.grid(True)
plt.savefig("regression_all_factors_cleaned.png")

# Bar plot of standardized coefficients (betas); exluded the intercept
predictors = X_all_cleaned.columns  # Use the column names from the predictors
plt.figure(figsize=(9, 6))
plt.barh(predictors, coefficients_cleaned[1:], align='center', alpha=0.8, color='skyblue')
plt.xlabel("Standardized Coefficient", fontsize=12)
plt.title("Standardized Coefficients (Betas) for Full Model", fontsize=14)
plt.axvline(x=0, color='black', linewidth=0.7)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("standardized_coefficients_bar_plot.png")
plt.show()



# QUESTION 9)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Prepare data for classification
X_class = cleaned_data[["Average_Rating"]].dropna()
y_class = cleaned_data.loc[X_class.index, "Received_Pepper"]

# Address class imbalance using stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=19128261, stratify=y_class
)

# Build the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict probabilities and classes
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]
y_pred_class = logistic_model.predict(X_test)

# Evaluate the model
auc_score = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred_class)
classification_rep = classification_report(y_test, y_pred_class)

# Calculate percentage of professors receiving pepper
percent_pepper = (cleaned_data["Received_Pepper"].sum() / len(cleaned_data))

# Calculate overall accuracy
overall_accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()

# Print Summary statistics
print(f"AUROC: {auc_score}")
print(f"Professors Receiving Pepper: {percent_pepper}")
print(f"Overall Accuracy: {overall_accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title("ROC Curve of Predicting Pepper from Average Rating")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_pepper_prediction.png")
plt.show()



# QUESTION 10)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Prepare data for classification using all available factors
X_all_class = cleaned_data[[
    "Average_Difficulty", "Number_of_Ratings", "Proportion_Take_Again", 
    "Online_Ratings", "Male", "Female"
]].dropna()  # Drop rows with missing data
y_all_class = cleaned_data.loc[X_all_class.index, "Received_Pepper"]

# Address class imbalance using stratified train-test split
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all_class, y_all_class, test_size=0.2, random_state=19128261, stratify=y_all_class
)

# Build the logistic regression model
logistic_model_all = LogisticRegression(max_iter=1000)
logistic_model_all.fit(X_train_all, y_train_all)

# Predict probabilities and classes
y_pred_prob_all = logistic_model_all.predict_proba(X_test_all)[:, 1]
y_pred_class_all = logistic_model_all.predict(X_test_all)

# Evaluate the model
auc_score_all = roc_auc_score(y_test_all, y_pred_prob_all)
conf_matrix_all = confusion_matrix(y_test_all, y_pred_class_all)
classification_rep_all = classification_report(y_test_all, y_pred_class_all)

# Calculate overall accuracy
overall_accuracy_all = (conf_matrix_all[0, 0] + conf_matrix_all[1, 1]) / conf_matrix_all.sum()

# Print Summary statistics
print(f"AUROC (All Factors): {auc_score_all}")
print(f"Overall Accuracy: {overall_accuracy_all}")
print("Confusion Matrix:")
print(conf_matrix_all)
print("Classification Report:")
print(classification_rep_all)

# Generate ROC curve
fpr_all, tpr_all, thresholds_all = roc_curve(y_test_all, y_pred_prob_all)
plt.figure(figsize=(8, 6))
plt.plot(fpr_all, tpr_all, label=f"ROC Curve (AUC = {auc_score_all:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title("ROC Curve of Predicting Pepper from All Factors")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_all_factors_pepper.png")
plt.show()



# EXTRA CREDIT)

# Load the cleaned dataset
cleaned_data_path = 'cleaned_rmp_data.csv'
cleaned_data = pd.read_csv(cleaned_data_path)

# Analyze the relationship between state and average rating
state_ratings = cleaned_data.groupby("State")["Average_Rating"].mean().sort_values(ascending=False)

# Calculate descriptive statistics
highest_rated_state = state_ratings.idxmax()
highest_rating = state_ratings.max()
lowest_rated_state = state_ratings.idxmin()
lowest_rating = state_ratings.min()

# Print Summary statistics
print(state_ratings.describe())
print(f"Highest-rated state: {highest_rated_state} ({highest_rating})")
print(f"Lowest-rated state: {lowest_rated_state} ({lowest_rating})")

# Plot the top 10 states with the highest average ratings
plt.figure(figsize=(10, 6))
state_ratings.head(10).plot(kind="bar", color="skyblue")
plt.title("Distribution of Top 10 States with Highest Average Ratings")
plt.xlabel("State")
plt.ylabel("Average Rating")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("top_states_avg_rating.png")
plt.show()
