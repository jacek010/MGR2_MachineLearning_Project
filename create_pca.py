import json
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tokens_dir = 'tokens'
tokens_file = 'tokens_full.json'

def load_from_json(filename):
    # Load data from a JSON file
    with open(f'{tokens_dir}/{filename}', 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    # Load the tokens from the JSON file
    tokens_data = load_from_json(tokens_file)
    
    # Create a DataFrame from the tokens
    tokens_list = []
    for item in tokens_data['data']:
        tokens_list.append(item['tokens'])
    
    tokens_df = pd.DataFrame(tokens_list).fillna(0)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tokens_df)
    
    # Add PCA results to the DataFrame
    tokens_df['pca-one'] = pca_result[:, 0]
    tokens_df['pca-two'] = pca_result[:, 1]
    
    # Plot PCA results
    plt.figure(figsize=(16,10))
    plt.scatter(tokens_df['pca-one'], tokens_df['pca-two'], c=tokens_df.index, cmap='viridis')
    plt.colorbar()
    plt.xlabel('PCA One')
    plt.ylabel('PCA Two')
    plt.title('PCA of Tokens')
    plt.show()
