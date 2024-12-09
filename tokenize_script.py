import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'llm_prepared_datasets'
input_file = f'{input_dir}/2018_llm_llama3_2_3000_new_explains_export.csv'

tokens_dir = 'tokens'
tokens_file = 'tokens_full.json'

def save_to_json(data, filename):
    # Save data to a JSON file
    data = {"input_file": input_file, 
            "data": data
            }
    if not os.path.exists(tokens_dir):
        os.makedirs(tokens_dir)
    with open(f'{tokens_dir}/{filename}', 'w') as f:
        json.dump(data, f)

def plot_pca(y_col, tfidf_vectorizer_vectors):
    
    pca = PCA(n_components=2)
    y = LabelEncoder().fit_transform(y_col)
    x_ = pca.fit_transform(tfidf_vectorizer_vectors.toarray(), y)
    x_ = x_[np.array(y_col)>-400]
    # a_max=np.argmax(pca.explained_variance_ratio_)
    # print(f"a_max: {a_max}")
    # print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    # print(tfidf_vectorizer.get_feature_names_out()[a_max])
    # print(f"Noise variance: {pca.noise_variance_}")
    # Plot PCA results
    plt.figure(figsize=(8,5))
    plt.scatter(x_[:, 0], x_[:, 1], c=np.array(y_col)[np.array(y_col)>-100], cmap='viridis', s=4)
    plt.colorbar()
    plt.xlabel('PCA One')
    plt.ylabel('PCA Two')
    plt.title('PCA of Tokens')
    plt.show()

def plot_token_importance(tfidf_vectorizer, tfidf_vectors, top_n=20):
    # Sum up the TF-IDF values for each token
    sums = tfidf_vectors.sum(axis=0)
    
    # Get the feature names (tokens)
    tokens = tfidf_vectorizer.get_feature_names_out()
    
    # Create a DataFrame with tokens and their corresponding TF-IDF sums
    token_importance = pd.DataFrame(sums.T, index=tokens, columns=["importance"]).sort_values(by="importance", ascending=False)
    
    # Plot the top N tokens
    plt.figure(figsize=(10, 6))
    token_importance.head(top_n).plot(kind='bar')
    plt.title(f'Top {top_n} Important Tokens')
    plt.xlabel('Tokens')
    plt.ylabel('Importance')
    plt.show()

if __name__ == '__main__':
    # Read the input CSV file into a DataFrame
    input_df = pd.read_csv(input_file).dropna(subset=["llm_delay_explain", "ARR_DELAY"])
    pred_col = input_df["llm_delay_explain"]
    delay_col = input_df["ARR_DELAY"]
    
    # Remove rows with NaN values
    pred_col = pred_col.dropna().tolist()
    delay_col = delay_col.tolist()
    #print(pred_col[0:5])
    
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(use_idf=True) 

    # Fit and transform the text data (SPARSE MATRIX)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(pred_col)
    
    # Plot PCA 
    plot_pca(delay_col, tfidf_vectorizer_vectors)
    
    # Plot token importance
    plot_token_importance(tfidf_vectorizer, tfidf_vectorizer_vectors, top_n=100)
    pca = PCA(n_components=2)
    y = LabelEncoder().fit_transform(delay_col)
    x_ = pca.fit_transform(tfidf_vectorizer_vectors.toarray(), y)
    x_ = x_[np.array(delay_col)>-100]
    # a_max=np.argmax(pca.explained_variance_ratio_)
    # print(f"a_max: {a_max}")
    # print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    # print(tfidf_vectorizer.get_feature_names_out()[a_max])
    # print(f"Noise variance: {pca.noise_variance_}")
    # Plot PCA results
    plt.figure(figsize=(16,10))
    plt.scatter(x_[:, 0], x_[:, 1], c=np.array(delay_col)[np.array(delay_col)>-100], cmap='viridis', s=4)
    plt.colorbar()
    plt.xlabel('PCA One')
    plt.ylabel('PCA Two')
    plt.title('PCA of Tokens')
    plt.show()
    exit()
    
    # Fit and transform the text data
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(pred_col)
    
    pca = PCA(n_components=2)
    y = LabelEncoder().fit_transform(delay_col)
    x_ = pca.fit_transform(tfidf_vectorizer_vectors.toarray(), y)
    x_ = x_[np.array(delay_col)>-100]
    # a_max=np.argmax(pca.explained_variance_ratio_)
    # print(f"a_max: {a_max}")
    # print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    # print(tfidf_vectorizer.get_feature_names_out()[a_max])
    # print(f"Noise variance: {pca.noise_variance_}")
    # Plot PCA results
    plt.figure(figsize=(16,10))
    plt.scatter(x_[:, 0], x_[:, 1], c=np.array(delay_col)[np.array(delay_col)>-100], cmap='viridis', s=4)
    plt.colorbar()
    plt.xlabel('PCA One')
    plt.ylabel('PCA Two')
    plt.title('PCA of Tokens')
    plt.show()
    exit()
    
    # Get the TF-IDF vector for the first document
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[1] 

    # Create a DataFrame with TF-IDF values
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names_out(), columns=["tfidf"]) 
    df = df.sort_values(by=["tfidf"], ascending=False)
    
    print(f"HEAD: {df.head(50)}")
    
    tokens = []
    for i in range(len(pred_col)):
        index_vector_tfidfvectorizer = tfidf_vectorizer_vectors[i]
        row = {"index": i,
               "text": pred_col[i],
               "tokens": {
                token: index_vector_tfidfvectorizer[0, tfidf_vectorizer.vocabulary_[token.lower()]]
                for token in sorted(pred_col[i].split(), key=lambda x: index_vector_tfidfvectorizer[0, tfidf_vectorizer.vocabulary_.get(x.lower(), 0)], reverse=True)
                if token.lower() in tfidf_vectorizer.vocabulary_
            }
        }
        tokens.append(row)
    
    # Save the tokens to a JSON file
    save_to_json(tokens, tokens_file)
