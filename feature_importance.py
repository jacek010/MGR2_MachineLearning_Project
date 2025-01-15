# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from matplotlib import pyplot
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

imputer = SimpleImputer(strategy='mean')
subdf = pd.DataFrame(imputer.fit_transform(subdf), columns=subdf.columns)

# define dataset
# X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
X = subdf.drop(columns=["ARR_DELAY"])
y = subdf["ARR_DELAY"]

X_cols = X.columns

# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_

print(np.argsort(importance))
# summarize feature importance
for i,v in enumerate(importance):
	# print('Feature: %0d, Score: %.5f' % (i,v))
    print(f"{i} Feature: {X_cols[i]}, Score: {v}")
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()