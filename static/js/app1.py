import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
from flask import Flask, render_template, request, session
from sklearn.neighbors import KNeighborsClassifier

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Save initial data
data = {"users": []}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)

def write_json(new_data, filename='DATA.json'):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["users"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

# Load datasets
df_tr = pd.read_csv('Training.csv')
df_tt = pd.read_csv('Testing.csv')

symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])

all_symp_col = list(df_tr.columns[:-1])

def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow')

all_symp = [clean_symp(sym) for sym in (all_symp_col)]

def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)

all_symp_pr = [preprocess(sym) for sym in all_symp]

col_dict = dict(zip(all_symp_pr, all_symp_col))

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
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a

def permutations(s):
    permutations = list(itertools.permutations(s))
    return [' '.join(permutation) for permutation in permutations]

def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symp_pr:
                return sym
    return False

def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None

def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return ptr, None

from nltk.wsd import lesk

def WSD(word, context):
    sens = lesk(context, word)
    return sens

def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))

def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim

def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))

def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)

def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a

def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis

def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

# Retrain the KNN model with current scikit-learn version
X_train = df_tr.iloc[:, :-1]  # Features
y_train = df_tr.iloc[:, -1]   # Labels

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

# Save the retrained model
joblib.dump(knn_clf, 'knn_new.pkl')

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

def getDescription():
    global description_list
    with open('Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            try:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
            except ValueError:
                print(f"Warning: Skipping row with invalid data: {row}")

def getprecautionDict():
    global precautionDictionary
    with open('precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

getDescription()
getSeverityDict()
getprecautionDict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_symptoms = request.form.get('symptoms')
    # Process input_symptoms and predict using the model
    # ...
    return render_template('result.html', prediction=predict)

if __name__ == '__main__':
    app.run(debug=True)
