

import gensim
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def embed_sentences(data, word2vec_limit = 50000 , NUM_WORDS=20000):   
    
    
    
    sentences = data[:,1]
    
    
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin', binary=True, limit=word2vec_limit)
    
    word_vectors = embedding_model.wv
    
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences)
     
    
    
    word_index = tokenizer.word_index
   
    embedding_weights = {key: embedding_model[word] if word in word_vectors.vocab else
                              np.random.uniform(-0.25, 0.25, word_vectors.vector_size)
                        for word, key in word_index.items()}
    
    embedding_weights[0] = np.zeros(word_vectors.vector_size)
   
    
    
    
    embedded_sentences = np.stack([np.stack([embedding_weights[token] for token in sentence]) for sentence in padded_sequences])
    
     
    
    input_output = np.array([])
    for i in range(len(data)):
        input_output = np.append(input_output,np.array([ embedded_sentences[i] , data[i,2] ]) )
        
    del embedding_model
    
    return input_output

def rand_embed_sentences(data, NUM_WORDS = None): 
    
    sentences = data[:,1]
    tokenizer = Tokenizer(num_words=NUM_WORDS)
   
    
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences)
    
    
    
    return padded_sequences, data[:,2]
    
if __name__ == "__main__":
    rand_embedded_sentences = rand_embed_sentences(np.array([[1, "hello!", 0.2], 
                                          [2,"cheese cake", 0.8]]))
    print(rand_embedded_sentences)
