import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import scipy.stats
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

input_dir = 'llm_prepared_datasets'
input_llm = 'llama3_2'
input_file = f'{input_dir}/2018_llm_{input_llm}_3000_new_explains_export.csv'

MODEL = 'GradientBoostingRegressor'

class ModelSelector:
    def __init__(self, model_name):
        if model_name == 'SVR':
            self.model = SVR(kernel='rbf')
        elif model_name == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor()
        elif model_name == 'RandomForestRegressor':
            self.model = RandomForestRegressor()
        elif model_name == 'LinearRegression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def get_model(self):
        return self.model
    

def extract_subdf(input_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_exclude = ["ARR_DELAY", "Unnamed: 27", "llm_delay_minutes", "llm_delay_explain"]  # Replace with actual column names to exclude
    columns_used_to_generate_llm = ["FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME", "DISTANCE", "CRS_ARR_TIME" ]
    subdf = input_df[columns_used_to_generate_llm]
    
    # Remove non-numeric columns
    # numeric_cols = subdf.select_dtypes(include=[np.number]).columns
    # subdf = subdf[numeric_cols]
    
    # Convert date column to numeric
    subdf["FL_DATE"] = pd.to_datetime(subdf["FL_DATE"]).map(pd.Timestamp.toordinal)
    
     # Encode categorical columns
    label_encoders = {}
    for column in ["OP_CARRIER", "ORIGIN", "DEST"]:
        le = LabelEncoder()
        subdf[column] = le.fit_transform(subdf[column])
        label_encoders[column] = le
    
    # Replace NaN values with the mean of the column
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    subdf = pd.DataFrame(imputer.fit_transform(subdf), columns=subdf.columns)
    
    return subdf

def sample(input_df: pd.DataFrame):
    df = extract_subdf(input_df)
    target = input_df["ARR_DELAY"]
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.3, shuffle=False, random_state=99
    )

    model = ModelSelector(MODEL).get_model()
    model.fit(X_train, y_train)

    print("\nSAMPLE DATA")
    print("Sample train score:", model.score(X_train, y_train))
    print("Sample test score:", model.score(X_test, y_test))

    # Cross-validation
    cv_scores = cross_val_score(model, df, target, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())
    
    return cv_scores

def vectorized(input_df: pd.DataFrame):
    text_data = input_df["llm_delay_explain"]
    target = input_df["ARR_DELAY"]
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(text_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.3, shuffle=True, random_state=99
    )

    model = ModelSelector(MODEL).get_model()
    model.fit(X_train, y_train)

    print("\nVECTORIZED DATA")
    print("Vectorized train score:", model.score(X_train, y_train))
    print("Vectorized test score:", model.score(X_test, y_test))

    # Cross-validation
    cv_scores = cross_val_score(model, X, target, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())
    
    return cv_scores


def count_mean_llm_delay_error(input_df: pd.DataFrame) -> dict:
    """
    Count mean error of predicting delay by llm using lower squares method and one extra method.

    Args:
        input_df (pd.DataFrame): DataFrame containing actual delays and predicted delays by llm.

    Returns:
        dict: Dictionary containing mean squared error and mean absolute error.
    """
    # Extract the relevant columns and convert to numeric
    actuals = pd.to_numeric(input_df["ARR_DELAY"], errors='coerce').values
    predictions = pd.to_numeric(input_df["llm_delay_minutes"], errors='coerce').values
    
    # Remove NaN values that were introduced by coercion
    mask = ~np.isnan(actuals) & ~np.isnan(predictions)
    actuals = actuals[mask]
    predictions = predictions[mask]
    
    # Mean Squared Error (MSE)
    mse = np.mean((predictions - actuals) ** 2)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - actuals))
    
    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae
    }
    
def success_llm_predictions(input_df: pd.DataFrame) -> dict:
    """
    Calculate the percentage of correct, overly optimistic, and overly pessimistic predictions.

    Args:
        input_df (pd.DataFrame): DataFrame containing actual delays and predicted delays by llm.

    Returns:
        dict: Dictionary containing percentages of correct, overly optimistic, and overly pessimistic predictions.
    """
    actuals = pd.to_numeric(input_df["ARR_DELAY"], errors='coerce').values
    predictions = pd.to_numeric(input_df["llm_delay_minutes"], errors='coerce').values
    
    # Remove NaN values that were introduced by coercion
    mask = ~np.isnan(actuals) & ~np.isnan(predictions)
    actuals = actuals[mask]
    predictions = predictions[mask]
    
    correct = np.sum(np.sign(actuals) == np.sign(predictions))
    overly_optimistic = np.sum((actuals > 0) & (predictions < 0))
    overly_pessimistic = np.sum((actuals < 0) & (predictions > 0))
    
    
    perfect = np.sum(actuals == predictions)
    less_than_5 = np.sum(np.abs(actuals - predictions) < 5)
    between_5_15 = np.sum((np.abs(actuals - predictions) >= 5) & (np.abs(actuals - predictions) < 15))
    between_15_30 = np.sum((np.abs(actuals - predictions) >= 15) & (np.abs(actuals - predictions) < 30))
    between_30_60 = np.sum((np.abs(actuals - predictions) >= 30) & (np.abs(actuals - predictions) < 60))
    between_60_120 = np.sum((np.abs(actuals - predictions) >= 60) & (np.abs(actuals - predictions) < 120))
    more_than_120 = np.sum(np.abs(actuals - predictions) >= 120)
        
    
    total = len(actuals)
    
    
    return {
        'correct_percentage': round((correct / total) * 100, 3),
        'overly_optimistic_percentage': round((overly_optimistic / total) * 100, 3),
        'overly_pessimistic_percentage': round((overly_pessimistic / total) * 100, 3),
    
        'perfect': round((perfect / total) * 100, 3),
        'less_than_5': round((less_than_5 / total) * 100, 3),
        'between_5_15': round((between_5_15 / total) * 100, 3),
        'between_15_30': round((between_15_30 / total) * 100, 3),
        'between_30_60': round((between_30_60 / total) * 100, 3),
        'between_60_120': round((between_60_120 / total) * 100, 3),
        'more_than_120': round((more_than_120 / total) * 100, 3)
    }


if __name__ == '__main__':
    input_df = pd.read_csv(input_file).dropna(subset=["llm_delay_explain", "ARR_DELAY"])
    input_df = input_df.fillna(0)
    
    print(f'LLM model: {input_llm} | Regression model:{MODEL}')
    sample_cv = sample(input_df)
    vector_cv = vectorized(input_df)
    
    errors = count_mean_llm_delay_error(input_df)
    print(errors)
    
    percentage_stats = success_llm_predictions(input_df)
    print(percentage_stats)

    ranksums = scipy.stats.ranksums(sample_cv, vector_cv)
    print(ranksums)