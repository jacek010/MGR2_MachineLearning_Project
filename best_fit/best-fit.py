# Data
import math
import nbformat
import numpy as np
import pandas as pd
from tqdm import tqdm

# Data Visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Data Processing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Linear Models
from rulefit import RuleFit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor

# Model Metrics
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# Model Interpretations
from interpret import show
import statsmodels.api as sm
from interpret.perf import ROC
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
from interpret.glassbox import ExplainableBoostingClassifier

# Dimensionality Reduction Methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Data file path
# file_path = "./Datasets/FlightDelays.csv"
file_path = "llm_prepared_datasets/2018_llm_gemma2_3000_new_explains_export.csv"

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Problem at hand
# Our task is to generate predictions for flights' delays and cancellaction rates based using ML algorithms based on accurate and informative data about those flights.
# Analyzing flight delays dataset

# Load data as a data frame
df = pd.read_csv(file_path)
df.drop(columns=['Unnamed: 27'], inplace=True)

# Quick look at the data frame
df.head()

# How many samples and features are the in dataset?
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# How many are there quantative and qualitive features?
df_numeric = df.select_dtypes(include='number')
df_numeric

# What features are informative?
df.nunique()
df_cv = df.select_dtypes(include='number').std() / df.select_dtypes(include='number').mean()
df_cv

# Data cleaning
# Remove bad features from the dataframe
df = df.loc[:, [f for f in df.columns if f not in df_cv or df_cv[f] > 0.4]]
df

null_num = df.isnull().sum()
null_num

df = df.loc[:, [f for f in df.columns if null_num[f] < df.shape[0] * 0.9]]
df

# Remove bad examples (rows) from the dataframe
df = df.dropna().reset_index()
df

df.isnull().sum()

# Data Preparation/Processing
# Focus on the most popular operating carrier
# Display the most common OP_CARRIER
df["OP_CARRIER"].value_counts()

# Filter out the most popular OP_CARRIER and drop the OP_CARRIER column
df = df[df["OP_CARRIER"] == "WN"].drop(columns=["OP_CARRIER"])
df

# Break the flight date by month and day of the week
# Flight date
df.FL_DATE

# Convert FL_DATE to datetime
dt_time = pd.to_datetime(df.FL_DATE)
dt_time

# Make "FL_MON" (month) and "FL_DOW" (day of the week) columns
df["FL_MON"] = dt_time.apply(lambda x: x.month)
df["FL_DOW"] = dt_time.apply(lambda x: x.dayofweek)
df = df.drop(columns=["FL_DATE"])
df

# Focus on the hub airports
# Find the hub airports
or_dest = (df["ORIGIN"].value_counts() + df["DEST"].value_counts()).sort_values(ascending=False)
freq = np.cumsum(or_dest) / or_dest.sum()
threshold = np.where(freq > 0.7)[0][0]
hub_airports = or_dest.index[:threshold+1].to_list()
hub_airports

# Make 'ORIGIN_HUB' and 'DEST_HUB' bool columns that will indicate that df.ORIGIN and df.DEST in hub_airports
df["ORIGIN_HUB"] = df["ORIGIN"].isin(hub_airports).astype(int)
df["DEST_HUB"] = df["DEST"].isin(hub_airports).astype(int)
df = df.drop(columns=["ORIGIN", "DEST", "OP_CARRIER_FL_NUM"])

df

# Focus on the carrier related delay issues
df = df.drop(columns=["DEP_DELAY", "ARR_DELAY", "WEATHER_DELAY", "NAS_DELAY", "LATE_AIRCRAFT_DELAY"])
df

# Inspect the feature impact of the target value
target = "CARRIER_DELAY"
plt.figure(figsize=(30, 40))
for i, col in enumerate(df.drop(columns=["index", "CARRIER_DELAY"]).columns):
    try:
        plt.subplot(3, 4, i + 1)
        plt.scatter(df[col], df[target])
        plt.title(col)
        print(f"Feature: {col}")
        print(df[[col, target]].dropna().head())
    except ValueError:
        break

df.drop(columns=["CANCELLED", "DIVERTED", "index"], inplace=True)
df

df = df.reset_index().drop(columns=["index"])
df

df = df.sample(int(df.shape[0] * 0.3))

# Split the df data into train and test
df = df.drop(columns=["llm_delay_explain"])
X = df.copy()
y = X.pop(target)
X = StandardScaler().fit_transform(X.to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train_labels, y_test_labels = y_train.apply(lambda y: y > 15), y_test.apply(lambda y: y > 15)

# Fitting models
# Generalized Regression Models
regression_models = {}

# **Linear Regression**
regression_models["linear"] = {"model": LinearRegression()}

# **Ridge Regression**
regression_models["ridge"] = {"model": RidgeCV()}

# **Decision Tree Regression**
regression_models["decision_tree"] = {"model": DecisionTreeRegressor()}

# **K-Nearest Neighbors Regression**
regression_models["knn"] = {"model": KNeighborsRegressor()}

# **Random Forest Regression**
regression_models["random_forest"] = {"model": RandomForestRegressor()}

# **Gradient Boosting**
regression_models["gradient_boosting"] = {"model": GradientBoostingRegressor()}

# Generalized Classification Models
classification_models = {}

# **Logistic Regression**
classification_models["logistic"] = {"model": LogisticRegression()}

# **Decision Tree**
classification_models["decision_tree"] = {"model": DecisionTreeClassifier()}

# **KNN Classifier**
classification_models["knn"] = {"model": KNeighborsClassifier()}

# **Naive Bayes**
classification_models["naive_bayes"] = {"model": GaussianNB()}

# **Random Forest**
classification_models["random_forest"] = {"model": RandomForestClassifier()}

# **Gradient Boosting**
classification_models["gradient_boosting"] = {"model": GradientBoostingClassifier()}

# Fit all models and make predictions
# Regression
for model in regression_models.keys():
    regression_models[model]["model"] = regression_models[model]["model"].fit(X_train, y_train)
    y_pred = regression_models[model]["model"].predict(X_test)
    regression_models[model]["mse"] = MSE(y_test, y_pred)
    regression_models[model]["r2"] = R2(y_test, y_pred)

# Classification
for model in classification_models.keys():
    classification_models[model]["model"] = classification_models[model]["model"].fit(X_train, y_train_labels)
    y_pred = classification_models[model]["model"].predict(X_test)
    classification_models[model]["precision"] = precision_score(y_pred, y_test_labels, zero_division=0)
    classification_models[model]["recall"] = recall_score(y_pred, y_test_labels)
    classification_models[model]["f1"] = f1_score(y_pred, y_test_labels)

# Visualize regression model metrics
regression_models = pd.DataFrame(regression_models)
regression_models

scores = regression_models.loc[["mse", "r2"]]
print(scores)
px.imshow(scores)

# Visualize classification model metrics
classification_models = pd.DataFrame(classification_models)
scores = classification_models.loc[["precision", "recall", "f1"]]
print(scores)
px.imshow(scores)

# Get the linear regression model
linear_regression = regression_models["linear"]["model"]

# Calculate feature importance
feature_importance = np.std(X_train, 0) * linear_regression.coef_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': df.drop(columns=[target]).columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("Intercept:", linear_regression.intercept_)
print("Feature Importance:")
print(feature_importance_df)

from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Add a constant to the model (intercept)
X_train_ols = sm.add_constant(X_train)

# Build and train the OLS model
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Report all stats
ols_summary = ols_model.summary(xname=['const'] + list(df.drop(columns=[target]).columns))
print(ols_summary)

# Ridge regression
ridge_regression = regression_models["ridge"]["model"]
ridge_feature_importance = np.std(X_train, 0) * ridge_regression.coef_

ridge_feature_importance_df = pd.DataFrame({
    'Feature': df.drop(columns=[target]).columns,
    'Importance': ridge_feature_importance
}).sort_values(by='Importance', ascending=False)

print("Ridge Regression Intercept:", ridge_regression.intercept_)
print("Ridge Regression Feature Importance:")
print(ridge_feature_importance_df)

# Logistic Regression
logistic_regression = classification_models["logistic"]["model"]
logistic_feature_importance = np.std(X_train, 0) * logistic_regression.coef_[0]

logistic_feature_importance_df = pd.DataFrame({
    'Feature': df.drop(columns=[target]).columns,
    'Importance': logistic_feature_importance
}).sort_values(by='Importance', ascending=False)

print("Logistic Regression Intercept:", logistic_regression.intercept_)
print("Logistic Regression Feature Importance:")
print(logistic_feature_importance_df)

# KNN
knn_model = classification_models["knn"]["model"]
distances, indices = knn_model.kneighbors(X_train)

# Calculate example-wise average neighbor distance
distances_mean = distances.mean(axis=1)

# Plot example-wise average neighbor distance to visualize anomalies
plt.figure(figsize=(10, 6))
plt.plot(distances_mean, marker='o', linestyle='none')
plt.title('Example-wise Average Neighbor Distance')
plt.xlabel('Example Index')
plt.ylabel('Average Neighbor Distance')
plt.show()

# Print the distances mean to the console
print("Example-wise Average Neighbor Distance:")
print(distances_mean)

# Calculate anomaly threshold using mu + 3*sigma rule
mu = distances_mean.mean()
sigma = distances_mean.std()
thr = mu + 3 * sigma

# Get the indices for anomalous examples in X_train
anomaly_indices = np.where(distances_mean > thr)[0]

# Select data for anomalies
anomalies = pd.DataFrame(X_train[anomaly_indices], columns=df.drop(columns=[target]).columns)

# Print anomalies to the console
print("Anomalous Examples:")
print(anomalies)

# Plot histograms for each feature of these anomalies
anomalies.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Anomalous Examples')
plt.show()

# Naive Bayes
naive_model = classification_models["naive_bayes"]["model"]
print(naive_model.theta_)

# Create a DataFrame for the theta values
theta_df = pd.DataFrame({
    'Feature': df.drop(columns=[target]).columns,
    'Theta': naive_model.theta_[0]
})

# Create a bar plot
fig = px.bar(theta_df, x='Feature', y='Theta', title='Naive Bayes Theta Values')
fig.show()

# Print theta values to the console
print("Naive Bayes Theta Values:")
print(theta_df)
