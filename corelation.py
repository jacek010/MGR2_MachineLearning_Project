import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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

input_dir = 'llm_prepared_datasets'
input_file = f'{input_dir}/2018_llm_gemma2_3000_new_explains_export.csv'

input_df = pd.read_csv(input_file)#.dropna(subset=["llm_delay_explain", "ARR_DELAY"])
subdf=extract_subdf(input_df)
# columns_to_exclude = ["Unnamed: 27", "llm_delay_minutes", "llm_delay_explain"]  # Replace with actual column names to exclude
# subdf = input_df.drop(columns=columns_to_exclude)

# # Remove non-numeric columns
# numeric_cols = subdf.select_dtypes(include=[np.number]).columns
# subdf = subdf[numeric_cols]

correlation = subdf.corr()

print(correlation)