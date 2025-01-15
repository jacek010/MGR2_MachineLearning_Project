import pandas as pd
import numpy as np



input_dir = 'llm_prepared_datasets'
input_file = f'{input_dir}/2018_llm_gemma2_3000_new_explains_export.csv'

input_df = pd.read_csv(input_file).dropna(subset=["llm_delay_explain", "ARR_DELAY"])

columns_to_exclude = ["Unnamed: 27", "llm_delay_minutes", "llm_delay_explain"]  # Replace with actual column names to exclude
subdf = input_df.drop(columns=columns_to_exclude)

# Remove non-numeric columns
numeric_cols = subdf.select_dtypes(include=[np.number]).columns
subdf = subdf[numeric_cols]

correlation = subdf.corr()

print(correlation)