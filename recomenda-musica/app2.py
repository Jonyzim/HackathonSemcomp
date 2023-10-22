import pandas as pd
import cohere
from sklearn.decomposition import PCA
import numpy as np
# Specify the path to your CSV file and the column name with text data
csv_file_path = 'spotify.csv'
text_column_name = 'lyrics'  # Replace with the actual column name
text_prompt="""Futuristic, technological, mystery, intrigue"""


# Initialize the Cohere client with your API key
co = cohere.Client('L41u8TnPpclKHjF0jJCxjD0SZ8O5yFQOaXoTibRL')
# Function to embed text data
def embed_text(text_data, model='small'):
    response = co.embed(texts=text_data, model=model)
    return response

df = pd.read_csv(csv_file_path,nrows=4000)

mask = df['lyrics'].str.len() < 6000
df = df[mask]
#df = df[mask]
# Read data from a CSV file using Pandas
def read_csv_and_embed(csv_file_path, text_column_name):
    global df
    text_data = df[text_column_name].tolist()

    text_data.append(text_prompt)
    embeddings = embed_text(text_data)
    return embeddings


# Call the function to read the CSV and embed the text data
embeddings = read_csv_and_embed(csv_file_path, text_column_name)

artist_data = df['track_artist'].tolist()
name_data = df['track_name'].tolist()

text_data = df[text_column_name].tolist()
# Check if embeddings is empty or None
if embeddings is None or not embeddings:
    print("Embeddings are empty or None.")
else:
    # Combine the embeddings for each artist
    lyrics_embedding = np.vstack(embeddings)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=500)  # Specify the number of components you want
    reduced_embeddings = pca.fit_transform(lyrics_embedding).tolist()
    reduced_embeddings=lyrics_embedding.tolist()
    # print(sum(pca.explained_variance_ratio_))
    # Print the reduced embeddings
    # print(reduced_embeddings)
    prompt=reduced_embeddings[-1]
    prompt=np.array(prompt)
    reduced_embeddings.pop()
    mindist = 100000
    i=0
    min_id=-1
    dist_list=[]
    for emb in reduced_embeddings:
        dist = np.linalg.norm(np.array(emb)-prompt)
        dist_list.append([dist,name_data[i],i,artist_data[i]])
        i+=1
    dist_list=sorted(dist_list, key=lambda x: x[0])
    songs=[]
    for d in dist_list[0:200]:
        #print(round(d[0],2),d[1])
        d[0]=round(d[0],2)
        songs.append(text_data[d[2]])

    #print(text_data[dist_list[0][2]])

    query = "What are the lyrics with mood most similar to '"+text_prompt+"'"
    print(query)
    results = co.rerank(query=query, documents=songs, top_n=10, model='rerank-english-v2.0') # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
    for idx, r in enumerate(results):
        #print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
        print(f"Song: {r.index} | {dist_list[r.index][1]} - {dist_list[r.index][3]}")
        #if idx ==0:
        #    print(f"Document: {r.document['text']}")
        #print(f"Relevance Score: {r.relevance_score:.2f}")
        #print("\n")
    

