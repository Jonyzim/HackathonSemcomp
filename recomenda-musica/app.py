import pandas as pd
import cohere
from sklearn.decomposition import PCA
import numpy as np
# Specify the path to your CSV file and the column name with text data
csv_file_path = 'lyrics-data.csv'
text_column_name = 'Lyric'  # Replace with the actual column name
text_prompt="Girafinha"
# Initialize the Cohere client with your API key
co = cohere.Client('L41u8TnPpclKHjF0jJCxjD0SZ8O5yFQOaXoTibRL')
# Function to embed text data
def embed_text(text_data, model='small'):
    response = co.embed(texts=text_data, model=model)
    return response

df = pd.read_csv(csv_file_path, nrows=1000)
# Read data from a CSV file using Pandas
def read_csv_and_embed(csv_file_path, text_column_name):
    text_data = df[text_column_name].tolist()
    text_data.append(text_prompt)
    embeddings = embed_text(text_data)
    return embeddings


# Call the function to read the CSV and embed the text data
embeddings = read_csv_and_embed(csv_file_path, text_column_name)


name_data = df['SLink'].tolist()
# Check if embeddings is empty or None
if embeddings is None or not embeddings:
    print("Embeddings are empty or None.")
else:
    # Combine the embeddings for each artist
    artist_embeddings = np.vstack(embeddings)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=2)  # Specify the number of components you want
    reduced_embeddings = pca.fit_transform(artist_embeddings).tolist()

    # Print the reduced embeddings
    # print(reduced_embeddings)
    prompt=reduced_embeddings[-1]
    reduced_embeddings.pop()
    mindist = 100000
    i=0
    min_id=-1
    dist_list=[]
    for emb in reduced_embeddings:
        dx = emb[0] - prompt[0]
        dy= emb[1] - prompt[1]
        dx *= dx
        dy *= dy 
        dist = dx + dy 
        dist_list.append([dist,name_data[i]])
        i+=1
    dist_list=sorted(dist_list, key=lambda x: x[0])
    print(dist_list[0:5])