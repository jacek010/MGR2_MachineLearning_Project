# Data
import math
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

# Load data as a data frame
df = pd.read_csv(file_path)
df.drop(columns=['Unnamed: 27'], inplace=True)

# Quick look at the data frame
df.head()

df_numeric = df.select_dtypes(include='number')
df_numeric