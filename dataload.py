import numpy as np
import pickle
import os
import re
from word_embedding import embed_sentences
from nltk import tokenize
from rouge import Rouge


def splitAndSanitizeIntoSentences(text):
    sentences = []
    v = text[0]
    subtext = v.text
    sentences = subtext.split(".")
    return sentences, len(sentences)


def parsePerdocs(path):
    f = open(path, "r")
    fullText = f.read().replace("\n", " ")
    f.close()
    
    
    summaries = {} # { docID : summary }
    sumIndex = fullText.find("DOCREF=")
   
    while sumIndex != -1:
        docID = fullText[sumIndex + 8:fullText.find("\"", sumIndex + 9)]
        
        startSum = fullText.find(">", sumIndex)
        endSum = fullText.find("</SUM>", sumIndex)

        text = fullText[startSum + 1:endSum]
        text = text.replace("<P>", " ")
        text = text.replace("</P>", " ")

        summaries[docID] = text

        sumIndex = fullText.find("DOCREF=", endSum) 
    
    for k in summaries.keys():
        summaries[k] = tokenize.sent_tokenize(summaries[k])

    return summaries
        
def extractText(path):
    f = open(path, "r")

    fullText = f.read().replace("\n", " ")
    f.close()        
    sentences = ""
    textIndex = fullText.find("<TEXT>")

    while textIndex != -1: 
        sentences += fullText[textIndex + 6 : fullText.find("</TEXT>", textIndex) ]
        textIndex = fullText.find("<TEXT>", textIndex + 1)

    
    sentences = sentences.replace("<P>", " ")
    sentences = sentences.replace("</P>", " ")
    
    sentences = sentences.replace(";", " ")

    return tokenize.sent_tokenize(sentences)

def _countMatchingTestData(sentences, summaries):
    size = 0
    hit = 0
    hitsize = 0 
    for s in sentences.keys():
        if s in summaries:
            hit += 1
            hitsize += len(sentences[s])
        size += len(sentences[s]) 
    return size, hit, hitsize

def _createEmbeddedTestData(sentences, summaries):

    size, hit, hitsize = _countMatchingTestData(sentences, summaries)

    test_data = []
    count = 0
    max_size = 0
    
    documents_over_190 = 0
    sentences_over_190 = 0
    sentences_removed = 0
    over_190 = False

    for s in sentences.keys():
        arr = np.ones((len(sentences[s]), 3), dtype=object) 
        arr[:,0] = "dummy"
        arr[:,1] = np.array(sentences[s])
        embedding = embed_sentences(arr)
        embedding = embedding[0::2]
        
        for e in embedding:
            if len(e) > max_size:
                max_size = len(e)
            if len(e) > 190:
                sentences_over_190 += 1
                over_190 = True
        if over_190:
            documents_over_190 += 1
            over_190 = False
            count -= len(sentences[s])
            sentences_removed += len(sentences[s])
            continue
        
        count += len(sentences[s])
        test_data.append((np.array(sentences[s]), np.array(embedding), np.array(summaries[s])))
        print("Finished", count, "of", size,"sentences --", count/size,"%", end='\r')
    return test_data
    
def loadTestData(dataRoot):
    
    sentences = {}
    summaries = {}   

    test_data = []

    
    raw_docs = dataRoot + "/docs/"
    walker = os.walk(raw_docs)
    for x in walker:
        path = x[0]
        dirs = x[1]
        files = x[2]    
    
        if len(dirs) != 0:
            continue
    
        for f in files:
            print("file:", path + "/" + f, end="\r") 
            sentences[f] = extractText(path + "/" + f)
        
    
    raw_summaries = dataRoot + "/summaries/"
    walker = os.walk(raw_summaries)
    for x in walker:
        path = x[0]
        dirs = x[1]
        files = x[2]
        
        if len(dirs) != 0:
            continue 

        for f in files:
            print("summary file:", path + "/" + f, end="\r") 
            tmpSummaries = parsePerdocs(path + "/" + f)
            for k in tmpSummaries.keys():
                summaries[k] = tmpSummaries[k]

   
    size, hit, hitsize = _countMatchingTestData(sentences, summaries)   

    embedded = _createEmbeddedTestData(sentences, summaries)
    return embedded
    
           
      

def _calculateNumberOfSentences(summaries, data):
    
    incorrect = 0
    totalSentences = 0
    for k in data.keys():
        if k not in summaries:
            print(" key not found in summaries", k)
            incorrect += len(data[k])
            continue
        totalSentences += len(data[k])
    return totalSentences
      
 
def _packageInNumpyArray(summaries, data, saliency):
    
    
    totalSentences = _calculateNumberOfSentences(summaries, data)
    cind = 0
    seen = 0
    skipped = 0
    parsed = 0
    nx3output = np.zeros((totalSentences, 3), dtype=object)
    for k in data.keys():
        if k not in summaries.keys():
            continue
        seen += 1
        sentences = data[k]
        summary = np.array(summaries[k])
        for s in sentences:
            nx3output[cind, 0] = k
            nx3output[cind, 1] = s
            try:
                nx3output[cind, 2] = saliency(np.array([s]), summary) 
                parsed += 1
                print(" ---- totalSentences:", totalSentences, "cind:", cind)
            except Exception as e:
                skipped += 1
                print("ERROR: Skipping sentence:", s)
                nx3output[cind, 2] = -1
            cind += 1
    print("parsed:", parsed, "Skipped:", skipped)
    return nx3output




def loadDUC(dataRoot, summarySize, saliency):
    
    
    
    rawData = {}
    rawSummaries = {}
    
    
    totalSentences = 0 

    
    walker = os.walk(dataRoot)
    for x in walker:
        
       
        path = x[0]
        dirs = x[1]
        files = x[2]

        
        if len(dirs) == 0: 
            if "perdocs" not in files:
                
                for f in files:
                    
                    try:
                        text = extractText(path + "/" + f)
                        totalSentences += len(text)
                        rawData[f] = text
                    except Exception as e:          
                        print("  ***", path + "/" + f) 
                        print(e)
            else:
                summaries = parsePerdocs(path + "/perdocs")
                for k in summaries.keys():
                    rawSummaries[k] = summaries[k]
    nx3output = _packageInNumpyArray(rawSummaries, rawData, saliency)
    return nx3output


def loadFromPickle(fileName):
    f = open(fileName, "rb")
    data = pickle.load(f)
    f.close()
    return data 

def dummy(sentence, summary):
    if sentence in summary:
        return 1
    return 0

def main():
    r = Rouge()
    testdata = loadTestData("../data/test_subset")
    
    print(testdata)
    

if __name__ == "__main__":
    main()
