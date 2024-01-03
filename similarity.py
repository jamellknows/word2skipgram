from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def similarity(word1, word2):
    
# Encode sentences

    embeddings1 = model.encode(word1, convert_to_tensor=True)
    embeddings2 = model.encode(word2, convert_to_tensor=True)

# Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    return cosine_similarity.item()
