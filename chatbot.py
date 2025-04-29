import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data and models
with open('dieseas.json', 'r') as f:
    intents = json.load(f)
intents 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load the KNN model (uncomment if model file exists)
knn=joblib.load('knn.pkl')

# Preprocess a given sentencei
def preprocess_sent(sent):
    t = nltk.word_tokenize(sent)
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in t if w not in set(stopwords.words('english')) and w.isalpha()])

# Create a bag of words representation
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

from sklearn.preprocessing import LabelEncoder
# Predict possible symptoms from a given sentence
def predictSym(sym,vocab,app_tag,df):
    label_encode = LabelEncoder()
    df['prognosis'] = label_encode.fit_transform(df['prognosis'])
    print(df)
    sym=preprocess_sent(sym)
    bow=np.array(bag_of_words(sym,vocab))
    print(bow)
    print(type(bow))
    res=cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
    print(res)
    order=np.argsort(res)[::-1].tolist()
    possym=[]
    
    app_tag_lenth=len(app_tag)
    for i in order:
        if i < app_tag_lenth:
            if app_tag[i].replace('_', ' ') in sym:
                return app_tag[i], 1
            if app_tag[i] not in possym and res[i] != 0:
                possym.append(app_tag[i])
    return possym, 0

# Convert symptoms to one-hot vectors
def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        if sym in all_sym:
            l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_sym)

# Check if all elements of small are in big
def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a

# Return possible diseases based on symptoms
def possible_Disease(l):
    poss_dis = []
    for dis in set(Disease):
        if contains(l, symVONDisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis

# Return symptoms associated with a given disease
def symVONDisease(df, Disease):
    ddf = df[df.prognosis == Disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

# Clean symptom names
def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace('yellowing', 'yellow')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

x_train=df_tr.iloc[:,:-1]
x_test=df.tt.iloc[:,:-1]
y_train=df_tr.iloc[:,-1]
y_test=df_tt.iloc[:,-1]

dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)

print(classification_report(y_test,knn_clf.predict(x_test)))


# Get user information

def getInfo():
    print("YOUR NAME \n\t\t\t", end="=>")
    name = input("")
    print("hello", name)
    return str(name)

# Global dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Load descriptions from CSV
def getDescription():
    global description_list
    with open('description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

# Load severity data from CSV
def getSeverityDict():
    global severityDictionary
    with open('severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

# Load precautions from CSV
def getprecautionDict():
    global precautionDictionary
    with open('precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

# Calculate condition severity
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary.get(item, 0)
    if (sum * days) / len(exp) > 13:
        return 1
    print("It might not be that bad but you should take precautions")
    return 0



# Initialize dictionaries
getSeverityDict()
getprecautionDict()
getDescription()

# Load training data
df_tr = pd.read_csv('Training.csv')
vocab = list(df_tr.columns)
Disease = df_tr.iloc[:, 1].tolist()
all_sym_col = list(df_tr.columns[:-1])
all_sym = [clean_symp(sym) for sym in all_sym_col]
app_tag = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        app_tag.append(tag)

# Main symptom processing function
def main_sp(name):
    print("Hi Mr/Ms " + name + ", can you describe your main symptom? \n\t\t\t\t\t\t", end="=>")
    sym1 = input("")
    psym1,find = predictSym(sym1, vocab, app_tag,df_tr)
    if find == 1:
        sym1 = psym1
    else:
        i = 0
        while i < len(psym1):
            print('Do you experience ' + psym1[i].replace('_', ' ') + '?')
            rep = input("")
            if rep.lower() == 'yes':
                sym1 = psym1[i]
                break
            else:
                i += 1

    print("Is there any other symptom, Mr/Ms " + name + "? \n\t\t\t\t\t\t", end="=>")
    sym2 = input("")
    psym2, find = predictSym(sym2, vocab, app_tag,df_tr)
    if find == 1:
        sym2 = psym2
    else:
        i = 0
        while i < len(psym2):
            print('Do you experience ' + psym2[i].replace('_', ' ') + '?')
            rep = input("")
            if rep.lower() == 'yes':
                sym2 = psym2[i]
                break
            else:
                i += 1
    
    # Create patient symptom list
    all_sym = [sym1, sym2]
    # Predict possible Disease
    Disease = possible_Disease(all_sym)
    stop = False
    print("Are you experiencing any of the following symptoms?")
    for dis in Disease:
        if not stop:
            for sym in symVONDisease(df_tr, dis):
                if sym not in all_sym:
                    print(clean_symp(sym) + ' ?')
                    while True:
                        inp = input("").lower()
                        if inp in ["yes", "no"]:
                            break
                        else:
                            print("Provide proper answers i.e. (yes/no): ", end="")
                    if inp == "yes":
                        all_sym.append(sym)
                        dise = possible_Disease(all_sym)
                        if len(dise) == 1:
                            stop = True
                            break
                    else:
                        continue
    return knn.predict(OHV(all_sym, all_sym_col)), all_sym

# Chat function
def chat_sp():
    a = True
    while a:
        name = getInfo()
        result, sym = main_sp(name)
        if result is None:
            ans3 = input("Can you specify more what you feel or tap 'q' to stop the conversation: ")
            if ans3.lower() == "q":
                a = False
            else:
                continue
        else:
            print("You may have " + result[0])
            print(description_list.get(result[0], "No description available."))
            an = input("How many days do you feel those symptoms? ")
            if calc_condition(sym, int(an)) == 1:
                print("You should take the consultation from a doctor.")
            else:
                print('Take the following precautions: ')
                for e in precautionDictionary.get(result[0], []):
                    print(e)
            print("Do you need another medical consultation (yes or no)? ")
            ans = input("").lower()
            if ans != "yes":
                a = False
                print("!!!!! Thanks for using our application !!!!!")

if __name__ == '__main__':
    chat_sp()