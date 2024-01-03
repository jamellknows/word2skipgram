from transformers import BertTokenizer, BertModel
from bert_serving.client import BertClient
import numpy as np

def get_bert_embeddings(text, model, tokenizer):
    # Tokenize the text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    # Get BERT embeddings
    inputs = tokenizer.encode_plus(text, return_tensors="pt", add_special_tokens=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    return tokens, embeddings

def bert_similarity(word1, word2, model, tokenizer, bc):
    _, emb1 = get_bert_embeddings(word1, model, tokenizer)
    _, emb2 = get_bert_embeddings(word2, model, tokenizer)

    # Reshape embeddings to 2D arrays
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)

    # Calculate cosine similarity
    similarity_score = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return similarity_score[0, 0]

# Example usage:
def run(word1, word2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    bc = BertClient()
    similarity_score = bert_similarity(word1, word2, model, tokenizer, bc)
    print(f"Similarity in meaning between '{word1}' and '{word2}': {similarity_score}")
    bc.close()
    return similarity_score
