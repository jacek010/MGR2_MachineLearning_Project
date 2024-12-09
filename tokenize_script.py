import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

input_dir = 'llm_prepared_datasets'
input_file = f'{input_dir}/2018_llm_llama3_2_3000_new_explains_export.csv'

tokens_dir = 'tokens'
tokens_file = 'tokens.json'

def save_to_json(data, filename):
    # Save data to a JSON file
    data = {"input_file": input_file, 
            "data": data
            }
    if not os.path.exists(tokens_dir):
        os.makedirs(tokens_dir)
    with open(f'{tokens_dir}/{filename}', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    # Read the input CSV file into a DataFrame
    input_df = pd.read_csv(input_file)
    pred_col = input_df["llm_delay_explain"]
    
    # Remove rows with NaN values
    pred_col = pred_col.dropna().tolist()
    print(pred_col[0:5])
    
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(use_idf=True) 

    # Fit and transform the text data
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(pred_col)
    
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
