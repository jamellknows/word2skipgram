import numpy as np 
import string 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
import math
import random
from gensim.models import KeyedVectors
import re
from similarity import similarity
import pandas as pd 
def softmax(x,h,N):
    """Compute softmax values for each set of scores in x."""
    target_size = np.prod(x.shape)
    source_size = np.prod(h.shape)
    add_size = target_size - source_size
    if add_size == 0:
        return np.sqrt(N - ((N/2)-x) - ((N/2)-h))
    elif add_size < 0:
        h = h[:len(x)]
        return np.sqrt(N - ((N/2)-x) - ((N/2)-h))
    source_end = x[-add_size:]
    h1 = np.append(h, source_end)
    h1 = h1.reshape(-1,1)
    return np.sqrt(N - ((N/2)-x) - ((N/2)-h1))
  
class word2vec(object):
    def __init__(self):
        self.N = 10
        self.X_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.001
        self.words = []
        self.word_index = {}
  
    def initialize(self,V,data):
        self.V = V
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
          
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
  
      
    def feed_forward(self,X):
        self.h = np.dot(self.W.T,X).reshape(self.N,1)
        self.u = np.dot(self.W1.T,self.h)
        self.y = softmax(self.u, self.h, self.N)  
        return self.y
          
    def backpropagate(self,x,t):
        e = self.y - np.asarray(t).reshape(self.V,1)
        # e.shape is V x 1
        dLdW1 = np.dot(self.h,e.T)
        X = np.array(x).reshape(self.V,1)
        dLdW = np.dot(X, np.dot(self.W1,e).T)
        self.W1 = self.W1 - self.alpha*dLdW1
        self.W = self.W - self.alpha*dLdW
          
    def train(self,epochs):
        for x in range(1,epochs):        
            self.loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                self.backpropagate(self.X_train[j],self.y_train[j])
                C = 0
                for m in range(self.V):
                    if(self.y_train[j][m]):
                        self.loss += -1*self.u[m][0]
                        C += 1
                self.loss += C*np.log(np.sum(np.exp(self.u)))
            print("epoch ",x, " loss = ",self.loss)
            self.alpha *= 1/( (1+self.alpha*x) )
             
    def predict(self,word,number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
              
            top_context_words = []
            for k in sorted(output,reverse=True):
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=number_of_predictions):
                    break
      
            return top_context_words
        else:
            print("Word not found in dictionary")
def preprocessing(corpus):
    stop_words = set(stopwords.words('english'))    
    training_data = []
    sentences = corpus.split(".")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence
                                     if word not in stop_words]
        x = [word.lower() for word in x]
        training_data.append(x)
    return training_data
      
  
def prepare_data_for_training(sentences,w2v):
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
      
    #for i in range(len(words)):
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
             
            for j in range(i-w2v.window_size,i+w2v.window_size):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word)
            w2v.y_train.append(context)
    w2v.initialize(V,data)
  
    return w2v.X_train,w2v.y_train   

def read_file_and_convert_to_array(file_path):
    # Initialize an empty array to store the lines
    lines_array = []

    # Read the file line by line and append each line to the array
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading or trailing whitespaces
            cleaned_line = line.strip()
            # Append the cleaned line to the array
            lines_array.append(cleaned_line)

    return lines_array

def select_random_word(sentence):
    # Split the sentence into words
    words = re.findall(r'\b\w+\b', sentence)
    words = [word for word in words if word != "the"]
    # Check if there are words in the sentence
    if words:
        # Select a random word
        random_word = random.choice(words)
        return random_word
    else:
        return "No words in the sentence"
    
def calculate_variance(numbers):
    n = len(numbers)
    
    # Calculate the mean
    mean = sum(numbers) / n
    
    # Calculate the squared differences from the mean
    squared_diff = [(x - mean) ** 2 for x in numbers]
    
    # Calculate the variance
    variance = sum(squared_diff) / n
    
    return variance
    
# corpus = ""
# corpus += "The earth revolves around the sun. The moon revolves around the earth"
def skipgram_square():
    output_file_path = 'results.txt'
    epochs = 1000
    file_path = "corpus.txt"
    corpus = read_file_and_convert_to_array(file_path)
    index = 0
    avg_scores = []
    columns2 = ["Context Word 1", "Context Word 2", "Context Word 3", "Context Word 4", "Context Word 5"]
    context = np.array([columns2])

    context = np.vstack(context)
    for i in range(0, len(corpus)-2):
        columns = ["Center Word", "Context Words Array", "Average Similarity Score", "Variance"]
        training_data = preprocessing(corpus[i])
        w2v = word2vec()
        prepare_data_for_training(training_data, w2v)
        w2v.train(epochs)
        rand_word = select_random_word(str(training_data))
        print(w2v.predict(str(rand_word),5))
        results = w2v.predict(str(rand_word),5)
        avg_sim_score = 0
        sim_score_array = []
        print(rand_word)
        results_good = True
        if results is None:
            sim_score_array = 0
        else:
            if results_good is True:
                for j in range(0,len(results)):
                    similarity_score = similarity(results[j],rand_word)
                    # print(f"Similarity between {rand_word} and {results[j]}: {similarity_score}")
                    sim_score_array.append(similarity_score)

            avg_sim_score = sum(sim_score_array)/len(sim_score_array)
            variance = calculate_variance(sim_score_array)
            # print(sim_score_array)
            data = [f"{rand_word}", f"{results}", f"{avg_sim_score}", f"{variance}"]
            data_context = [results[0], results[1], results[2], results[3], results[4]]
            data_context = np.array(data_context)
            sim_score_array = np.array(sim_score_array)
            df2 = pd.DataFrame(columns = columns2)
            avg_scores.append(data)
            context = np.vstack((context, data_context))
            context = np.vstack((context, sim_score_array))
    np.savetxt("square_context_results.csv", context, delimiter=',', fmt='%s')
    df = pd.DataFrame(columns = columns)
    for i in range(0,len(avg_scores)):
        df.loc[i] = avg_scores[i]

    df.to_csv(f"square_center_results.csv", index="True")
    return 

skipgram_square()
    
    
    
