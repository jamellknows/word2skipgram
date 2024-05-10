import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from collections import Counter
import heapq
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
import math


# Define the SkipGram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, softmax_method=None, forward_method='linear'):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.forward_method = forward_method
        self.softmax_method = softmax_method

    def forward(self, target):
        if self.forward_method == 'linear':
            embedded = self.embeddings(target)
            output = self.linear(embedded)
        elif self.forward_method == 'quad':
            embedded = self.embeddings(target)
            output = torch.pow(embedded, 2) + embedded 
            output = self.linear(embedded)
        elif self.forward_method == 'cross':
            output = embedded * embedded 
            output = self.linear(output)
        elif self.forward_method == 'square':
            output = torch.sqrt(2 * embedded - (torch.full_like(embedded, torch.max(embedded)) - embedded))
            output = self.linear(output)
        elif self.forward_method == 'angle':
            output = torch.cos(embedded)
            output = self.linear(output)
        elif self.forward_method == '9':
            output = 9 * torch.tan(embedded) - 9 + torch.cos(embedded) + torch.sin(embedded)
            output = self.linear(output)
        else:
            raise ValueError("Unknown forward method:", self.forward_method)

        if self.softmax_method == 'softmax':
            output = F.softmax(output, dim=-1)
        elif self.softmax_method == 'cosinemax':
            output = torch.cos(output)
        elif self.softmax_method == '9max':
            output = 9 * torch.tan(embedded) - 9 + torch.cos(embedded) + torch.sin(embedded)
        elif self.softmax_method == 'squaremax':
            output = torch.sqrt(2 * embedded - (torch.full_like(embedded, torch.max(embedded)) - embedded))
        elif self.softmax_method == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.softmax_method == 'dirlecht':
            output = torch.distributed.dirichlet.Dirichlet(0.5 * torch.exp(output)).rsample()
        elif self.softmax_method == 'logistic':
            output = torch.sigmoid(output * 0.5 + 2)
        elif self.softmax_method == 'beta':
            output = torch.distributions.beta.Beta(0.5, 2).cdf(torch.sigmoid(output))
        elif self.softmax_method == 'probability':
            output = torch.exp(output)/torch.sum(torch.exp(output))
        return output
           
# Example training data
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps"
]

# Tokenize the corpus
tokens = [word for sentence in corpus for word in sentence.split()]
word_counts = Counter(tokens)
vocab = {word: index for index, (word, _) in enumerate(word_counts.most_common())}
reverse_vocab = {index: word for word, index in vocab.items()}

# Hyperparameters
embedding_dim = 50
window_size = 2
learning_rate = 0.001
epochs = 100



# Training loop (skipped for brevity)
# Function to find top context words for a sentence


# SIMILARITY METHODS 


#sine method 

def sine_similarity(a, b):
    # Convert input lists to PyTorch tensors
    vector1 = torch.tensor(a, requires_grad=True)
    vector2 = torch.tensor(b, requires_grad=True)

    # Compute dot product
    dot_product = torch.sum(vector1 * vector2, dim=1)

    # Compute magnitudes
    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)

    # Compute cosine of the angle between vectors
    cosine = dot_product / (magnitude1 * magnitude2)

    # Compute sine similarity
    angle_radians = torch.acos(cosine)
    sine_similarity = torch.sin(angle_radians)

    return sine_similarity.item()

def tangent_similarity(a, b):
    dot_product = torch.sum(a * b, dim=1)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    angle = torch.acos(dot_product / (norm_a * norm_b))
    similarity = torch.tan(angle) / (norm_a * norm_b)
    return similarity


#euclidean method 
def euclidean_similarity(a,b):
    
    return euclidean(a,b)

def manhatten_similarity(a,b):
    distance_np = cityblock(a.detach().numpy().flatten(), b.detach().numpy().flatten())
    distance_tensor = torch.tensor(distance_np, dtype=torch.float32)

    return distance_tensor

#jaccard method 
def jaccard_similarity(a,b):
    # Convert vectors to sets of indices where the value is non-zero
    set1 = set(index for index, value in enumerate(vector1) if value != 0)
    set2 = set(index for index, value in enumerate(vector2) if value != 0)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union != 0 else 0 

# RUNNING 

def find_top_context_words_cosine(sentence, model, vocab, reverse_vocab, top_k=3):
    words = sentence.split()
    top_context_words = {}
    for word in words:
        if word in vocab:
            word_index = vocab[word]
            input_tensor = torch.LongTensor([word_index])
            word_embedding = model.embeddings(input_tensor)
            similarity_scores = []
            for context_word, context_index in vocab.items():
                if context_word != word:
                    context_tensor = torch.LongTensor([context_index])
                    context_embedding = model.embeddings(context_tensor)
                    cosine_similarity = nn.functional.cosine_similarity(word_embedding, context_embedding, dim=-1)
                    similarity_scores.append((cosine_similarity, context_word))
            top_context_words[word] = heapq.nlargest(top_k, similarity_scores)
    return top_context_words

# sine method 

def find_top_context_words_sine(sentence, model, vocab, reverse_vocab, top_k=3):
    words = sentence.split()
    top_context_words = {}
    for word in words:
        if word in vocab:
            word_index = vocab[word]
            input_tensor = torch.LongTensor([word_index])
            word_embedding = model.embeddings(input_tensor)
            similarity_scores = []
            for context_word, context_index in vocab.items():
                if context_word != word:
                    context_tensor = torch.LongTensor([context_index])
                    context_embedding = model.embeddings(context_tensor)
                    sine_similarity_score = sine_similarity(word_embedding, context_embedding)
                    similarity_scores.append((sine_similarity_score, context_word))
            top_context_words[word] = heapq.nlargest(top_k, similarity_scores)
    return top_context_words

# tangent method 
def find_top_context_words_tangent(sentence, model, vocab, reverse_vocab, top_k=3):
    words = sentence.split()
    top_context_words = {}
    for word in words:
        if word in vocab:
            word_index = vocab[word]
            input_tensor = torch.LongTensor([word_index])
            word_embedding = model.embeddings(input_tensor)
            similarity_scores = []
            for context_word, context_index in vocab.items():
                if context_word != word:
                    context_tensor = torch.LongTensor([context_index])
                    context_embedding = model.embeddings(context_tensor)
                    tangent_similarity_score = tangent_similarity(word_embedding, context_embedding)
                    similarity_scores.append((tangent_similarity_score, context_word))
            top_context_words[word] = heapq.nlargest(top_k, similarity_scores)
    return top_context_words

# manhattan method 
def find_top_context_words_manhatten(sentence, model, vocab, reverse_vocab, top_k=3):
    words = sentence.split()
    top_context_words = {}
    for word in words:
        if word in vocab:
            word_index = vocab[word]
            input_tensor = torch.LongTensor([word_index])
            word_embedding = model.embeddings(input_tensor)
            similarity_scores = []
            for context_word, context_index in vocab.items():
                if context_word != word:
                    context_tensor = torch.LongTensor([context_index])
                    context_embedding = model.embeddings(context_tensor)
                    manhatten_similarity_score = manhatten_similarity(word_embedding, context_embedding)
                    similarity_scores.append((manhatten_similarity_score, context_word))
            top_context_words[word] = heapq.nlargest(top_k, similarity_scores)
    return top_context_words

# euclidean method 

def find_top_context_words_euclidean(sentence, model, vocab, reverse_vocab, top_k=3):
    words = sentence.split()
    top_context_words = {}
    for word in words:
        if word in vocab:
            word_index = vocab[word]
            input_tensor = torch.LongTensor([word_index])
            word_embedding = model.embeddings(input_tensor)
            similarity_scores = []
            for context_word, context_index in vocab.items():
                if context_word != word:
                    context_tensor = torch.LongTensor([context_index])
                    context_embedding = model.embeddings(context_tensor)
                    euclidean_similarity_score = euclidean_similarity(word_embedding, context_embedding)
                    similarity_scores.append((euclidean_similarity_score, context_word))
            top_context_words[word] = heapq.nlargest(top_k, similarity_scores)
    return top_context_words

# jaccard method 

def find_top_context_words_jaccard(sentence, model, vocab, reverse_vocab, top_k=3):
    words = sentence.split()
    top_context_words = {}
    for word in words:
        if word in vocab:
            word_index = vocab[word]
            input_tensor = torch.LongTensor([word_index])
            word_embedding = model.embeddings(input_tensor)
            similarity_scores = []
            for context_word, context_index in vocab.items():
                if context_word != word:
                    context_tensor = torch.LongTensor([context_index])
                    context_embedding = model.embeddings(context_tensor)
                    jaccard_similarity_score = jaccard_similarity(word_embedding, context_embedding)
                    similarity_scores.append((jaccard_similarity_score, context_word))
            top_context_words[word] = heapq.nlargest(top_k, similarity_scores)
    return top_context_words


# Example usage
# sentence = "the lazy dog sleeps"
# top_context_words = find_top_context_words_cosine(sentence, model, vocab, reverse_vocab)
# print('\n cosine similarity method')
# for word, context_words in top_context_words.items():
#     print(f"Top context words for '{word}': {context_words}")


# print('\n tangent similarity method')
# top_context_words = find_top_context_words_tangent(sentence, model, vocab, reverse_vocab)

# for word, context_words in top_context_words.items():
#     print(f"Top context words for '{word}': {context_words}")
    
    
# firstly get it to work with simple word context 
# need to find a dataset where the proteins are labeled as names, reaction pathways [what proteins it interacts with and effect on or with protein] 


#can write a softmax or evaluation function 


# use this to write a new model series -- this model does not include a softmax function, what I will do is use it to compare proteins in a sequence 
# I need a list of amino acids -- amino acid similarity -- or amino acid openers 
# change the feed forward, similarity method, add a softmax for comaprision

# compare softmax to no softmax 
# compare types of softmax 
# 
if __name__ == '__main__':
    print('Similarity methods are sine, cosine, tangent, manhatten, euclidean, jaccard or s,c,t,m,e,j \n')
    similarity_method = input('Choose a similarity method \n')
    print('Softmax methods are softmax, none \n')
    softmax_method = input('Choose a softmax method \n ')
    print('Forward methods are linear, cross, square, angle, quadratic, \n')
    forward_method = input('Choose a forward method \n')
    #put in defintions for forward model and tangent method to use when running the thing 
    model = SkipGram(len(vocab), embedding_dim, softmax_method, forward_method)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sentence = "the lazy dog sleeps"
    
    def config_run(similarity_method):
        # group by similarity methods 
        if similarity_method == 'sine':
            print(f'Running top context words sine with {softmax_method} and {forward_method}')
            top_context_words = find_top_context_words_sine(sentence, model, vocab, reverse_vocab)
            print('\n sine similarity method \n')
            for word, context_words in top_context_words.items():
                print(f"Top context words for '{word}': {context_words}")
            
        elif similarity_method == 'cosine':
            print(f'Running top context words cosine with {softmax_method} and {forward_method}')
            top_context_words = find_top_context_words_cosine(sentence, model, vocab, reverse_vocab)
            print('\n cosine similarity method \n')
            for word, context_words in top_context_words.items():
                print(f"Top context words for '{word}': {context_words}")
            
        elif similarity_method == 'tangent':
            print(f'Running top context words tangent with {softmax_method} and {forward_method}')
            top_context_words = find_top_context_words_cosine(sentence, model, vocab, reverse_vocab)
            print('\n tangent similarity method \n')
            for word, context_words in top_context_words.items():
                print(f"Top context words for '{word}': {context_words}")
            
        elif similarity_method == 'manhatten':
            print('Running top context words manhatten')
            top_context_words = find_top_context_words_manhatten(sentence, model, vocab, reverse_vocab)
            print('\n tangent similarity method \n')
            for word, context_words in top_context_words.items():
                print(f"Top context words for '{word}': {context_words}")
            
            
        elif similarity_method == 'euclidean':
            print('Running top context words euclidean')
            top_context_words = find_top_context_words_euclidean(sentence, model, vocab, reverse_vocab)
            print('\n tangent similarity method \n')
            for word, context_words in top_context_words.items():
                print(f"Top context words for '{word}': {context_words}")

        elif similarity_method == 'jaccard':
            print('Running top context words jaccard')  
            top_context_words = find_top_context_words_jaccard(sentence, model, vocab, reverse_vocab)
            print('\n tangent similarity method \n')
            for word, context_words in top_context_words.items():
                print(f"Top context words for '{word}': {context_words}")      
        else:
            return 'similarity method not found'
        
    config_run(similarity_method)
        
        #forward model selection is used to iteratively train a model on what features to use for it's ml programme. 
        
        #ensure that it is using a forward method of selection - complete all forward methods and ensure that it is using the correct softmax method 
        # softmax study also needed, include square_max and 9 max, softmax is currently exponential, do not raise anything to an exponent find a new way to 
        # sigmoid, normalisation, dirlecht, logistic, beta, pobability calibration, square, 9, I would develop 3 methods cosine, sine and tangent probab conversions
    
    