# WE ARE TRAINING THE NLP...... SEMANTIC ANALYSIS

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#tokenizing
from nltk.tokenize import sent_tokenize, word_tokenize
data="Hello, i am very happy to meet you. I created this course for you. Good by!"
doc1='Computer science is the study of computers and computing concepts. It includes both hardware and software, as well as networking and the Internet'
sentences=sent_tokenize(data) #divise en phrase
print(sentences)
words=word_tokenize(data)#divise en word token
print(words)

#cleaning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
data="Hello, i am very happy to meet you. I created this course for you. Good by!"
word_tokens = [word.lower() for word in word_tokenize(data)]
data_clean = [word for word in word_tokens if (not word in set(stopwords.words('english')) and  word.isalpha())]
print(data_clean)

#lemmatization:a text pre-processing technique used in natural language processing (NLP) models to break a word down to its root meaning to identify similarities. 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
data=" I am a Chat-bot.I am also called as Medical Assistant. I give you advise about your health and predicts the disease you are having based on the symptom.I will also give you menta health councilling and also doctor appointment scheduling"
words=word_tokenize(data)
words=[lemmatizer.lemmatize(word.lower())for word in words if(not word in set(stopwords.words('english'))and word.isalpha())]
print(words)

#POS TAGGING assigning a grammatical category (such as noun, verb, adjective, etc.) to each word in a sentence. 
from nltk.tokenize import word_tokenize
data=" I am a Chat-bot.I am also called as Medical Assistant. I give you advise about your health and predicts the disease you are having based on the symptom.I will also give you menta health councilling and also doctor appointment scheduling"
words=word_tokenize(data)
print(nltk.pos_tag(words))#words
from nltk.tokenize import sent_tokenize
data=" I am a Chat-bot.I am also called as Medical Assistant. I give you advise about your health and predicts the disease you are having based on the symptom.I will also give you menta health councilling and also doctor appointment scheduling"
sentences=sent_tokenize(data)
list=[]
for sentence in sentences:
    list.append(word_tokenize(sentence))
    print(nltk.pos_tag_sents(list))

    #brown:identifying foreign words
    nltk.download('brown')
    from nltk.corpus import brown
    from nltk.tag import UnigramTagger #Unigram Tagger: For determining the Part of Speech tag
    from nltk.tokenize import word_tokenize
    brown_tagged_sents=brown.tagged_sents(categories='news')
    size=int(len(brown_tagged_sents)*0.9)
    train_sents=brown_tagged_sents[:size]
    test_sents=brown_tagged_sents[size:]
    Unigram_tagger=nltk.UnigramTagger(train_sents)
    print(Unigram_tagger.evaluate(test_sents))
    data=" I am a Chat-bot.I am also called as Medical Assistant. I give you advise about your health and predicts the disease you are having based on the symptom.I will also give you menta health councilling and also doctor appointment scheduling"
    print(Unigram_tagger.tag(word_tokenize(data)))
    brown_tagged_sents = brown.tagged_sents(categories='news')
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    bigram_tagger = nltk.BigramTagger(train_sents)
    print(bigram_tagger.evaluate(test_sents))
    data=" I am a Chat-bot.I am also called as Medical Assistant. I give you advise about your health and predicts the disease you are having based on the symptom.I will also give you menta health councilling and also doctor appointment scheduling"
    print(bigram_tagger.tag(word_tokenize(data.lower())))

# initializing single noun
    brown_tagged_sents=brown.tagged_sents(categories='news')
    size=int(len(brown_tagged_sents)*0.9)
    train_sents=brown_tagged_sents[:size]
    test_sents=brown_tagged_sents[size:]
    t0=nltk.DefaultTagger('NN')# assigns same tag to each token
    t1=nltk.UnigramTagger(train_sents, backoff=t0)#taking 1 word at a time
    t2=nltk.UnigramTagger(test_sents, backoff=t1)#taking 2 words at a time
    print(t2.evaluate(test_sents))
    data=" I am a Chat-bot.I am also called as Medical Assistant. I give you advise about your health and predicts the disease you are having based on the symptom.I will also give you menta health councilling and also doctor appointment scheduling"
    print(t2.tag(word_tokenize(data)))

    #NAMED ENTITY RECOGNITION
    data="hiii I am Krithika I am studing in ASC in Bangalore.I am a computer science student.I like Jungkook.He is handsome"
    words=word_tokenize(data)
    print(nltk.ne_chunk(nltk.pos_tag(words)))#ne_chunks:identifies name ,locations and org

    #word sense disambiguation (WSD):
    from nltk.corpus import wordnet
    for sens in wordnet.synsets('break'):#allows you to look up words in WordNet
        print('>>>',sens.definition())

#LESK ALGORITHM words used in a text are related to one another, and that this relationship can be seen in the definitions of the words and their meanings
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
context=word_tokenize("I've just finished the first step of the competition. I need a little break to catch my breath")
print(lesk(context,'break',).definition())

#PYTHON WSD
from pywsd.lesk import simple_lesk
sent = "I've just finished the first step of the competition. I need a little break to catch my breath"
ambiguous='break'
answer=simple_lesk(sent,ambiguous,pos='n')
print(answer)
print(answer.definition())

#SPACYprovides advanced capabilities to conduct natural language processing (NLP) on large volumes of text at high speed.
import spacy

# ONE-HOT-VECTOR: ensures that machine learning does not assume that higher numbers are more important.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
freq=CountVectorizer()
corpus=['I am Rene','I am beautiful','this is my first attempt']
corpus=[sent.lower()for sent in corpus]
corpus=freq.fit_transform(corpus)
onehot=Binarizer()
corpus=onehot.fit_transform(corpus.toarray())
print(corpus)

#BAG OF WORDS
vectorizer=CountVectorizer()
corpus=['I am Rene','I am beautiful','this is my first attempt']
corpus=[sent.lower()for sent in corpus]
X=vectorizer.fit_transform(corpus)
print(X.toarray()) # explicit matrix format
print(vectorizer.get_feature_names_out() ) #vocabulary as list of string
vectorizer.vocabulary_.get('document') #get column index of a specific term in the vocabulary
vectorizer.transform(['Something completely new.']).toarray()#apply the model to a new document

#TF-TDF:to assess the importance of words in a document relative to a collection of documents.
from sklearn.feature_extraction.text import TfidfVectorizer #used for converting a collection of raw documents into a matrix of TF-IDF features
corpus=['I am Rene','I am beautiful','this is my first attempt']
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())
#The tf-idf vector obtained will be normalized to obtain values between 0 and 1.

# To fetch ingo from google
import glob
files=glob.glob("www.google.com")
f_pointers=[open(files ,"r",encoding="utf8",errors='ignore')for file in sorted(files)]
corpus=[f.read()for f in f_pointers]
print(corpus)

#nlp=spacy.load('en_core_web_sm')
doc=[nlp(doc)for doc in corpus]
from spacy.lang.en.stop_words import STOP_WORDS
def preprocess(doc_nlp):
    d=[]
    for token in doc_nlp:
        if(not token.text in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_)
            return d
        preprocess(doc[0])

        #SIMILARITY MEASURES :EUCLIDEAN DISTANCE COSINE JACCARD
        from sklearn.metrics.pairwise import cosine_distances,euclidean_distances
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer=TfidfVectorizer()
        corpus=['I am Rene','I am beautiful','this is my first attempt']
        tfidf_matrix=tfidf_vectorizer.fit_transform(corpus)
        tfidf_matrix.shape
        euclidean_distances(tfidf_matrix[0],tfidf_matrix)

        #OHV:
        corpus=['I am Rene','I am beautiful','this is my first attempt']
        freq=CountVectorizer()
        c=[sent.lower() for sent in corpus]
        cor=freq.fit_transform(corpus)
        euclidean_distances(cor[0],cor)


from sklearn.metrics.pairwise import cosine_distances,euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
import time

    
def tfidf_simil(corpus,index_of_doc):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    #compute similarity for first sentence with rest of the sentences
    return (euclidean_distances(tfidf_matrix[index_of_doc],tfidf_matrix))

def ohv_simil(corpus,index_of_doc):
    vectorizer  = CountVectorizer()
    corpus=[ sent.lower() for sent in corpus]
    corpus = vectorizer.fit_transform(corpus)

    onehot = Binarizer()
    ohv = onehot.fit_transform(corpus.toarray())

    return (euclidean_distances(ohv[index_of_doc],ohv))

def bow_simil(corpus,index_of_doc):
    vectorizer = CountVectorizer(lowercase=True)
    corpus=[ sent.lower() for sent in corpus]
    X = vectorizer.fit_transform(corpus) #sparsy format
    bow = X.toarray()
    
    return (euclidean_distances(bow[index_of_doc],bow))
task1=corpus[::5]
task2=corpus[2::5]
task3=corpus[3::5]
task4=corpus[4::5]
task5=corpus[5::5]
from nltk.corpus import wordnet as wn 
computer_synsets = wn.synsets("medical") 
print("Computer sens in wordNet:")
i=0;
for sense in computer_synsets: 
    print(" \t Sens :", i)
    print(" \t\t Sens definition: "+sense.definition())
    lemmas = [l.name() for l in sense.lemmas()]
    print("\t\t Lemmas for sense :" +str(lemmas))
    i=i+1

from nltk.corpus import wordnet as wn 
import pandas as pd
import numpy as np
computer_synsets = wn.synsets("medical") 
device_synsets = wn.synsets("study") 
lch=[]
wup=[]
print(device_synsets)
for s1 in computer_synsets:
    for s2 in device_synsets:
        lch.append(s1.path_similarity(s2))
        wup.append(s1.wup_similarity(s2))

pd.DataFrame([lch,wup],["lch","wup"])



