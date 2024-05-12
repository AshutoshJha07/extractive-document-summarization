import numpy as np
from keras.models import load_model
from preprocessing.dataload import loadTestData
from preprocessing.rouge import Rouge

def dummy_rouge(sentence_arr, summary_arr, alpha=0.5):
    return np.random.rand()

def dummy_loadTestData():
    testing_data = [ [ np.array(["This sentence is important for doc0." ,
                                 "Such a sentence is irrelevent for doc 0."]), 
                       np.random.rand(2,5,300), 
                       np.array(["This sentence is important for doc0."]) ],
                     [ np.array(["Lol that sentence is awesome for do1." , 
                                 "No way, this is irrelevent"]), 
                       np.random.rand(2,5,300), 
                                np.array(["Lol that sentence is awesome for do1."]) ] ]
    return testing_data

def test(model, testing_data, batch_size = 128, upper_bound = 100, threshold = 1, metric = "ROUGE1"):
     
    evals = []
    
    for doc in testing_data: 
        sentences = doc[0]
        
        
            

        true_summary = doc[2]
        
        sentences_num = np.random.randint(len(sentences))
        predicted_summary = np.random.choice(sentences.tolist(), sentences_num, replace=False)        
        
        
        if metric == "ROUGE1" :
            N = 1
        elif metric == "ROUGE2":
            N = 0 
            
        evals.append(dummy_rouge( predicted_summary, true_summary, alpha = N))
        
    return np.mean(evals)
    

def main():
    model = load_model('model.h5')
    
    testing_data = loadTestData("./data/DUC2002_Summarization_Documents")
    
    rouge1_score = test(model, testing_data, upper_bound=100, metric = "ROUGE1")
    
    print("")
    print(rouge1_score)

if __name__ == "__main__":
    main()

