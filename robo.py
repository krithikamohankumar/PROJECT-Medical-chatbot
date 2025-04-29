import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn 
import csv
import json

data={"users":[]}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)

def write_json(new_data, filename='DATA.json'):
    with open(filename,'r+') as file:# First we load existing data into a dict.
        file_data = json.load(file)# Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)# Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)
 
df_tr=pd.read_csv('Training.csv')
print(df_tr.head())
print(df_tr.shape)
print(df_tr.iloc[-1])
df_tt=pd.read_csv('Testing.csv')
print(df_tt.head())
# Initialize empty lists for symptoms and diseases
symp = []
disease = []

# Iterate over each row in the DataFrame
for i in range(len(df_tr)):
    # Find columns where the value is 1, indicating the presence of a symptom
    row_symptoms = df_tr.columns[df_tr.iloc[i] == 1].tolist()
    # Append the list of symptoms to symp list
    symp.append(row_symptoms)
    # Append the corresponding disease to the disease list
    disease.append(df_tr.iloc[i, -1])

print(symp[:5])  # Print first 5 lists of symptoms
print(disease[:5])  # Print first 5 diseases

# Extract all symptom column names except the last column (assumed to be 'Disease' column)
all_symp_col = list(df_tr.columns[:-1])
# Function to clean symptom names
def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace('yellowing', 'yellow')
# Apply the cleaning function to all symptom column names
all_sym = [clean_symp(sym) for sym in all_symp_col]
# Output the cleaned symptom names to verify correctness
print(all_sym[:10])  # Print first 10 cleaned symptom names

from nltk.corpus import wordnet as wn
ohne_syns = []  # Symptoms without synonyms in WordNet
mit_syns = []   # Symptoms with synonyms in WordNet
# Iterate through the cleaned symptom names and categorize them
for sym in all_sym:
    if not wn.synsets(sym):
        ohne_syns.append(sym)
    else:
        mit_syns.append(sym)
# Output the counts to verify correctness
print(f"Symptoms with synonyms: {len(mit_syns)}")
print(f"Symptoms without synonyms: {len(ohne_syns)}")

from spacy.lang.en.stop_words import STOP_WORDS
import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if not token.text.lower() in STOP_WORDS and token.text.isalpha():
            d.append(token.lemma_.lower())
    return ' '.join(d)

def preprocess_sym(doc):
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)
print (preprocess("skin peeling"))
all_symp_pr=[preprocess_sym(sym) for sym in all_sym]#Preprocess the Symptom Names
col_dict = dict(zip(all_symp_pr, all_symp_col))#Create the Dictionary

#SYNTATIC SIMILARITY
def jaccard_set(str1, str2):
    list1=str1.split(' ')
    list2=str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

#The syntactic_similarity function calculates the Jaccard similarity between a target symptom (symp_t) and a corpus of symptoms
import numpy as np

def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    # Calculate Jaccard similarity between the target symptom and each symptom in the corpus
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    # Sort indices of similarities in descending order
    order = np.argsort(most_sim)[::-1].tolist()
    # Check for existence of symptoms in descending order of similarity
    for i in order:
        if DoesExist(corpus[i]):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    # If any possible symptoms were found, return them
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None
    
import itertools
# Returns all the subsets of this set. This is a generator.
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item

def sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if len(a[j])>len(a[i]):
                a[i],a[j]=a[j],a[i]
    a.pop()
    return a

def permutations(s):
    # Generate all permutations of the sequence s
    permutations = list(itertools.permutations(s))
    # Join each permutation into a single string and return as a list
    return [' '.join(permutation) for permutation in permutations]

def DoesExist(txt):#The DoesExist function is designed to check if a combination of symptoms (given as a text string txt) exists in a predefined list of preprocessed symptoms (all_symp_pr).
    # Split the input text into individual symptoms
    txt = txt.split(' ')
    # Generate all combinations (subsets) of the symptoms
    combinations = [x for x in powerset(txt)]
    sort(combinations)  # Assuming sort function sorts the combinations
    # Iterate through each combination of symptoms
    for comb in combinations:
        # Generate all permutations of the current combination
        for sym in permutations(comb):
            # Check if the permutation exists in the list of preprocessed symptoms
            if sym in all_symp_pr:
                return sym  # Return the matching symptom
    # If no matching symptom is found, return False
    return False
DoesExist('feel pain abdominal')
# Preprocess the symptom description
preprocess('my skin has some nodal eruptions')
# Calculate syntactic similarity with all_symp_pr
syntactic_similarity(preprocess('i experience pain in my abdominal'),all_symp_pr)
def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"  # Anchored pattern to match exactly inp
    regexp = re.compile(patt)  # Compile the regular expression pattern
    # Iterate through each item in dis_list
    for item in dis_list:
        # Check if the pattern matches the item
        if regexp.search(item):
            pred_list.append(item)  # Add matching items to pred_list
    # If matching items were found, return 1 and the list of matches
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return ptr, None  # Otherwise, return 0 and None
print(check_pattern('nail',all_symp_pr))

from nltk.wsd import lesk # Importing the lesk function from NLTK's Word Sense Disambiguation module.
from nltk.tokenize import word_tokenize
def WSD(word, context):
    sens=lesk(context, word)
    return sens

def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')  # Preprocess and tokenize doc1
    doc2_p = preprocess(doc2).split(' ')  # Preprocess and tokenize doc2
    score = 0
    # Iterate over all token pairs from doc1 and doc2
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)  # Perform WSD on token from doc1
            syn2 = WSD(tock2, doc2)  # Perform WSD on token from doc2
            # Check if both tokens were successfully disambiguated
            if syn1 is not None and syn2 is not None:
                x = syn1.path_similarity(syn2)  # Compute path similarity
                if x is not None and x > 0.25:  # Threshold for similarity
                    score += x  # Accumulate similarity score
    # Normalize score by dividing by the product of lengths of token lists
    return score / (len(doc1_p) * len(doc2_p))

def semantic_similarity(symp_t, corpus):#The semantic_similarity function you've defined aims to find the most semantically similar symptom (or text) from a corpus to a target symptom (symp_t) using the semanticD
    max_sim = 0
    most_sim = None
    # Iterate over each symptom in the corpus
    for symp in corpus:
        d = semanticD(symp_t, symp)  # Calculate semantic similarity between symp_t and symp
        # Update most_sim if current similarity score (d) is higher than max_sim
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim

print(semantic_similarity('puke',all_symp_pr))
semantic_similarity('puke',all_symp_pr)
print(all_symp_pr)

from itertools import chain
from nltk.corpus import wordnet
def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]# Extracts lemma names (actual synonyms) 
    lemmas = list(set(chain(*lemmas))) #removes duplicates, and converts it back to a list.
    # Iterate over each lemma and check semantic similarity
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)  # Check semantic similarity
        if res != 0:
            symp.append(sym1)  # Append the symptom if there's a positive similarity score
    return list(set(symp))  # Return unique list of suggested symptoms

print(suggest_syn('puke'))

import numpy as np
import pandas as pd

def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])  # Initialize a numpy array of zeros
    for sym in cl_sym:
        if sym in all_sym:
            l[0, all_sym.index(sym)] = 1  # Set 1 at the index corresponding to the symptom in all_sym
    return pd.DataFrame(l, columns=all_sym)  # Convert the array to a DataFrame with column names as all_sym

def contains(small, big):
    for i in small:#List of elements to check if they are all present in big
        if i not in big:#List (or any iterable) where small elements are checked for existence.
            return False
    return True

def possible_diseases(l):
    poss_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis
print(set(disease))

def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    if len(ddf) == 0:
        print(f"No data found for {disease}")
        return []  # Return empty list or handle appropriately
    # Assuming there's at least one row in ddf
    symptoms = ddf.columns[ddf.iloc[0] == 1].tolist()
    return symptoms
# Example usage:
# Replace df_tr and 'jaundice' with actual DataFrame and disease name
symptoms = symVONdisease(df_tr, 'jaundice')
print(symptoms)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Assuming X_train, X_test, y_train, y_test are already defined correctly
X_train = df_tr.iloc[:, :-1]
X_test = df_tt.iloc[:, :-1]
y_train = df_tr.iloc[:, -1]
y_test = df_tt.iloc[:, -1]

# Initialize KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_train)

# Initialize DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Evaluate KNeighborsClassifier
knn_predictions = knn_clf.predict(X_test)
print("KNeighborsClassifier Report:")
print(classification_report(y_test, knn_predictions))


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

getSeverityDict()
getprecautionDict()
getDescription()
print(severityDictionary)
def calc_condition(exp, days):
    total_sum = 0
    for item in exp:
        if item in severityDictionary.keys():
            total_sum += severityDictionary[item]
    if (total_sum * days) / len(exp) > 13:
        print("You should take the consultation from doctor.")
        return 1
    else:
        print("It might not be that bad but you should take precautions.")
        return 0
    
def getInfo():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("hello ",name)
    return str(name)

def related_sym(psym1):
    if len(psym1)==1:
        return psym1[0]
    print("searches related to input: ")
    for num,it in enumerate(psym1):
        print(num,")",clean_symp(it))
    if num!=0:
        print(f"Select the one you meant (0 - {num}):  ", end="")
        conf_inp = int(input(""))
    else:
        conf_inp=0
    disease_input=psym1[conf_inp]
    return disease_input

def main_sp(name,all_symp_col):
    #main Idea: At least two initial sympts to start with
    
    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Enter the main symptom you are experiencing Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym1 = input("")
    sym1=preprocess_sym(sym1)
    sim1,psym1=syntactic_similarity(sym1,all_symp_pr)
    if sim1==1:
        psym1=related_sym(psym1)
    
    #get the 2nd syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)

    print("Enter a second symptom you are experiencing Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    sym2=preprocess_sym(sym2)
    sim2,psym2=syntactic_similarity(sym2,all_symp_pr)
    if sim2==1:
        psym2=related_sym(psym2)
        
    #if check_pattern==0 no similar syntaxic symp1 or symp2 ->> try semantic similarity
    
    if sim1==0 or sim2==0:
        sim1,psym1=semantic_similarity(sym1,all_symp_pr)
        sim2,psym2=semantic_similarity(sym2,all_symp_pr)
        
        #if semantic sim syp1 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim1==0:
            sugg=suggest_syn(sym1)
            print('Are you experiencing any ')
            for res in sugg:
                print(res)
                inp=input('')
                if inp=="yes":
                    psym1=res
                    sim1=1
                    break
                
        #if semantic sim syp2 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim2==0:
            sugg=suggest_syn(sym2)
            for res in sugg:
                inp=input('Do you feel '+ res+" ?(yes or no) ")
                if inp=="yes":
                    psym2=res
                    sim2=1
                    break
        #if no syntaxic semantic and suggested sym found return None and ask for clarification

        if sim1==0 and sim2==0:
            return None,None
        else:
            # if at least one sym found ->> duplicate it and proceed
            if sim1==0:
                psym1=psym2
            if sim2==0:
                psym2=psym1
    #create patient symp list
    all_sym=[col_dict[psym1],col_dict[psym2]]
    #predict possible diseases
    diseases=possible_diseases(all_sym)
    stop=False
    print("Are you experiencing any ")
    for dis in diseases:
        print(diseases)
        if stop==False:
            for sym in symVONdisease(df_tr,dis):
                if sym not in all_sym:
                    print(clean_symp(sym)+' ?')
                    while True:
                        inp=input("")
                        if(inp=="yes" or inp=="no"):
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ",end="")
                    if inp=="yes":
                        all_sym.append(sym)
                        diseases=possible_diseases(all_sym)
                        if len(diseases)==1:
                            stop=True 
    return knn_clf.predict(OHV(all_sym,all_symp_col)),all_sym


def chat_sp():
    a=True
    while a:
        name=getInfo()
        result,sym=main_sp(name,all_symp_col)
        if result == None :
            ans3=input("can you specify more what you feel or tap q to stop the conversation")
            if ans3=="q":
                a=False
            else:
                continue

        else:
            print("you may have "+result[0])
            print(description_list[result[0]])
            an=input("how many day do you feel those symptoms ?")
            if calc_condition(sym,int(an))==1:
                print("you should take the consultation from doctor")
            else : 
                print('Take following precautions : ')
                for e in precautionDictionary[result[0]]:
                    print(e)
            print("do you need another medical consultation (yes or no)? ")
            ans=input()
            if ans!="yes":
                a=False
                print("!!!!! thanks for using ower application !!!!!! ")
print(df_tr.iloc[-1])
import joblib
knn_clf=joblib.load('knn.pkl')
print(symVONdisease(df_tr,"Jaundice"))
print(knn_clf.predict(OHV(['fatigue', 'weight_loss', 'itching','high_fever'],all_symp_col)) )
d=df_tr[df_tr.iloc[:,-1]=="Fungal infection"].sum(axis=0)
cl=df_tr.columns
pp=d!=0
print(cl[pp])
print(d[pp].drop('prognosis'))
chat_sp()
