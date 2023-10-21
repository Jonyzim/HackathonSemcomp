import cohere
co = cohere.Client('L41u8TnPpclKHjF0jJCxjD0SZ8O5yFQOaXoTibRL')

response = co.embed(
  texts=['hello', 'goodbye','pietra'],
  model='small',
)

# Reduce dimensionality using PCA
from sklearn.decomposition import PCA

# Function to return the principal components
def get_pc(arr,n):
  arr=np.vstack(arr)
  pca = PCA(n_components=n)
  embeds_transform = pca.fit_transform(arr)
  return embeds_transform

print(get_pc(response.embeddings[0],2))